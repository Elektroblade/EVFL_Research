import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset
from datasets import DatasetDict
import time
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader

INPUT_SIZE = 78
OUTPUT_SIZE = 3
HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 64
TASK_TYPE = "multiclass"
SEED = 42
BATCH_SIZE = 32

class DNN:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, learning_rate, activation, task_type="multiclass"):
        assert task_type in ["binary", "multiclass", "regression"], \
            f"Unsupported task_type: {task_type}"

        self.task_type = task_type
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation = activation
        if self.task_type == "binary":
            self.output_activation = sigmoid
        elif self.task_type == "multiclass":
            self.output_activation = softmax
        else:
            self.output_activation = self.activation
        
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for all layers randomly
        layer_sizes = [self.input_size] + [self.neurons_per_layer] * self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            bias_vector = np.random.randn(layer_sizes[i+1], 1)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def feedforward(self, x):
        activations = [x]
        weighted_sums = []

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            weighted_sums.append(z)

            if i == len(self.weights) - 1:
                a = self.output_activation(z)   # <-- output layer
            else:
                a = self.activation(z)

            activations.append(a)

        return activations, weighted_sums

    
    def backpropagation(self, x, y):
        gradients_w = [np.zeros(weight.shape) for weight in self.weights]
        gradients_b = [np.zeros(bias.shape) for bias in self.biases]
        
        # Feedforward
        activations = [x]
        weighted_sums = []
        
        for i in range(self.hidden_layers + 1):
            weighted_sum = np.dot(self.weights[i], activations[i]) + self.biases[i]
            weighted_sums.append(weighted_sum)
            activation_output = self.activation(weighted_sum)
            activations.append(activation_output)
        
        # Backpropagation
        if self.task_type in ["binary", "multiclass"]:
            delta = activations[-1] - y
        else:
            delta = (activations[-1] - y) * self.activation_derivative(weighted_sums[-1])
        gradients_w[-1] = np.dot(delta, activations[-2].T)
        gradients_b[-1] = delta
        
        for i in range(self.hidden_layers - 1, -1, -1):
            delta = np.dot(self.weights[i+1].T, delta) * self.activation_derivative(weighted_sums[i])
            gradients_w[i] = np.dot(delta, activations[i].T)
            gradients_b[i] = delta
        
        return gradients_w, gradients_b
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softmax(z):
        z = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


    
    def load_data(self, partition_id: int, num_partitions: int, batch_size: int, target_column: str = "Label",):
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)

            THIS_DIR = Path(__file__).resolve().parent
            PROJECT_ROOT = THIS_DIR.parents[1]   # evfl → dnn → EVFL_Research
            DATA_DIR = PROJECT_ROOT / "InSDN_DatasetCSV" / "InSDN_DatasetCSV"

            # Fail fast if path is wrong
            if not DATA_DIR.exists():
                raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

            metasploitable_df = pd.read_csv(DATA_DIR / "metasploitable-2.csv")
            normal_data_df = pd.read_csv(DATA_DIR / "Normal_data.csv")
            ovs_df = pd.read_csv(DATA_DIR / "OVS.csv")

            # Normalize
            metasploitable_df = normalize_columns(metasploitable_df)
            normal_data_df = normalize_columns(normal_data_df)
            ovs_df = normalize_columns(ovs_df)

            metasploitable_df = normalize_target_column(metasploitable_df, target_column)
            normal_data_df = normalize_target_column(normal_data_df, target_column)
            ovs_df = normalize_target_column(ovs_df, target_column)

            normal_data_df['Target Type'] = 'none'
            metasploitable_df['Target Type'] = 'Host'
            ovs_df['Target Type'] = 'SDN'
            insdn_data_df = pd.concat([metasploitable_df, normal_data_df, ovs_df], ignore_index=True)

            assert target_column in insdn_data_df.columns, \
                f"{target_column} not found in dataset"
            
            self.num_classes = insdn_data_df[target_column].nunique()

            insdn_data_df["target"] = insdn_data_df[target_column]

            hf_dataset = hf_dataset.remove_columns(
                [c for c in ["Label", "Target Type"]]
            )

            columns_without_labels = insdn_data_df.drop(columns=['Label', 'Target Type']).columns
            print("Columns in the dataset (excluding 'Label' and 'Target Type'):")
            print(list(columns_without_labels))

            hf_dataset = Dataset.from_pandas(insdn_data_df)
            hf_dataset = hf_dataset.class_encode_column("Label")
            dataset_dict = DatasetDict({
                "train": hf_dataset
            })

            fds = FederatedDataset(
                dataset=dataset_dict,
                partitioners={"train": partitioner},
            )
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(
            test_size=0.2,
            seed=SEED,
        )
        # Construct dataloaders
        partition_train_test = partition_train_test.with_transform(self.apply_transforms)
        g = torch.Generator()
        g.manual_seed(SEED)

        trainloader = DataLoader(
            partition_train_test["train"],
            batch_size=None,
            shuffle=True,
            generator=g,
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=None, shuffle=False)
        return trainloader, testloader
    
    def load_centralized_dataset(self, target_column = "Label"):
        # Load full dataset
        THIS_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = THIS_DIR.parents[1]   # evfl → dnn → EVFL_Research
        DATA_DIR = PROJECT_ROOT / "InSDN_DatasetCSV" / "InSDN_DatasetCSV"

        # Fail fast if path is wrong
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

        metasploitable_df = pd.read_csv(DATA_DIR / "metasploitable-2.csv")
        normal_data_df = pd.read_csv(DATA_DIR / "Normal_data.csv")
        ovs_df = pd.read_csv(DATA_DIR / "OVS.csv")
        
        # Normalize
        metasploitable_df = normalize_columns(metasploitable_df)
        normal_data_df = normalize_columns(normal_data_df)
        ovs_df = normalize_columns(ovs_df)

        metasploitable_df = normalize_target_column(metasploitable_df, target_column)
        normal_data_df = normalize_target_column(normal_data_df, target_column)
        ovs_df = normalize_target_column(ovs_df, target_column)

        normal_data_df['Target Type'] = 'none'
        metasploitable_df['Target Type'] = 'Host'
        ovs_df['Target Type'] = 'SDN'

        insdn_data_df = pd.concat(
            [metasploitable_df, normal_data_df, ovs_df],
            ignore_index=True
        )

        self.num_classes = insdn_data_df[target_column].nunique()

        insdn_data_df["target"] = insdn_data_df[target_column]

        hf_dataset = Dataset.from_pandas(insdn_data_df)
        hf_dataset = hf_dataset.class_encode_column(target_column)

        # Deterministic split
        dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)

        test_dataset = dataset["test"].with_transform(self.apply_transforms)

        return DataLoader(
            test_dataset,
            batch_size=None,
            shuffle=False,
        )
    
    def apply_transforms(self, batch):
        """
        Transform HF batch into NumPy arrays compatible with the custom DNN.
        """

        # 1️⃣ Extract target
        y = np.asarray(batch["Label"], dtype=np.int64)  # shape: (batch_size,)

        print(y.dtype, y[:5])

        # 2️⃣ Extract features (everything except target)
        feature_cols = [k for k in batch.keys() if k != "target"]
        X = np.stack([batch[col] for col in feature_cols], axis=1)
        # X shape: (batch_size, input_size)

        # 3️⃣ Transpose for DNN
        X = X.T  # (input_size, batch_size)

        # 4️⃣ Encode targets
        if self.task_type == "binary":
            y = y.reshape(1, -1)  # (1, batch_size)

        elif self.task_type == "multiclass":
            y_onehot = np.zeros((self.output_size, y.shape[0]))
            num_classes = self.num_classes  # or infer from model
            y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
            y_onehot[np.arange(y.shape[0]), y] = 1

            y = y_onehot  # (output_size, batch_size)

        else:  # regression
            y = y.reshape(self.output_size, -1)

        return {
            "x": X,
            "y": y,
        }
    
    def train(
        self,
        trainloader,
        max_epochs=50,
        convergence_threshold=1e-6,
    ):
        prev_loss = float("inf")

        for epoch in range(max_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in trainloader:
                x = batch["x"]  # (input_size, batch_size)
                y = batch["y"]  # (output_size, batch_size)

                # ---- Forward + backward ----
                gradients_w, gradients_b = self.backpropagation(x, y)

                # ---- Gradient descent update ----
                self.weights = [
                    w - self.learning_rate * gw
                    for w, gw in zip(self.weights, gradients_w)
                ]
                self.biases = [
                    b - self.learning_rate * gb
                    for b, gb in zip(self.biases, gradients_b)
                ]

                # ---- Loss computation ----
                activations, _ = self.feedforward(x)
                y_hat = activations[-1]

                if self.task_type == "binary":
                    # Binary cross-entropy
                    eps = 1e-9
                    loss = -np.mean(
                        y * np.log(y_hat + eps) +
                        (1 - y) * np.log(1 - y_hat + eps)
                    )

                elif self.task_type == "multiclass":
                    # Categorical cross-entropy
                    eps = 1e-9
                    loss = -np.mean(
                        np.sum(y * np.log(y_hat + eps), axis=0)
                    )

                else:
                    # Regression fallback (MSE)
                    loss = 0.5 * np.mean((y_hat - y) ** 2)

                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            # ---- Convergence check ----
            if abs(prev_loss - avg_loss) < convergence_threshold:
                print(f"Converged at epoch {epoch}")
                break

            prev_loss = avg_loss
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

        return prev_loss

    def predict(self, testloader):
        """
        Run inference on the entire testloader.

        Returns:
            predictions: np.ndarray (num_samples,)
            prediction_probs: np.ndarray (num_samples, num_classes)
            real_values: np.ndarray (num_samples,)
            avg_inference_time_ms: float
        """

        all_predictions = []
        all_prediction_probs = []
        all_real_values = []

        total_time = 0.0
        total_samples = 0

        for batch in testloader:
            x = batch["x"]  # (input_size, batch_size)
            y = batch["y"]  # (output_size, batch_size)

            batch_size = x.shape[0]

            # ---- Inference timing ----
            start = time.perf_counter()
            activations, _ = self.feedforward(x)
            end = time.perf_counter()

            total_time += (end - start)
            total_samples += batch_size

            y_hat = activations[-1]  # (output_size, batch_size)

            # ---- Predictions & probabilities ----
            if self.task_type == "binary":
                probs = y_hat.flatten()                 # (batch_size,)
                preds = (probs >= 0.5).astype(int)     # threshold
                real = y.flatten()

                all_prediction_probs.extend(
                    np.vstack([1 - probs, probs]).T    # (batch_size, 2)
                )

            elif self.task_type == "multiclass":
                probs = y_hat.T                         # (batch_size, num_classes)
                preds = np.argmax(probs, axis=1)

                real = np.argmax(y, axis=0)

                all_prediction_probs.extend(probs)

            else:  # regression fallback
                preds = y_hat.T
                probs = y_hat.T
                real = y.T

                all_prediction_probs.extend(probs)

            all_predictions.extend(preds)
            all_real_values.extend(real)

        avg_inference_time_ms = (total_time / total_samples) * 1000.0

        return (
            np.array(all_predictions),
            np.array(all_prediction_probs),
            np.array(all_real_values),
            avg_inference_time_ms,
        )

def dnn_to_arrays(model: DNN):
    return model.weights + model.biases


def arrays_to_dnn(model: DNN, arrays):
    # Convert ArrayRecord -> list
    arrays_list = list(arrays) if not isinstance(arrays, list) else arrays
    n_w = len(model.weights)
    model.weights = arrays_list[:n_w]
    model.biases = arrays_list[n_w:]

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def normalize_target_column(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in columns")

    # Convert to string → strip whitespace
    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.strip()
    )

    return df