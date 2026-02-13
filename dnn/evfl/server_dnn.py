import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, VerticalSizePartitioner
from datasets import Dataset
from datasets import DatasetDict
import time
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk

from evfl.dnn import (
    DNN,
    INPUT_SIZE,
    OUTPUT_SIZE,
    HIDDEN_LAYERS,
    NEURONS_PER_LAYER,
    PARTITION_SIZES,
    TASK_TYPE,
    DATASET_DIR,
    SEED,
    dnn_to_arrays,
    arrays_to_dnn,
    relu,
    relu_derivative
)

class ServerDNN:
    def __init__(
        self,
        num_clients,
        embedding_dim,
        output_size,
        hidden_layers,
        neurons_per_layer,
        learning_rate,
        activation,
        activation_derivative,
        task_type="multiclass"
    ):
        assert task_type in ["binary", "multiclass", "regression"]

        self.input_size = num_clients * embedding_dim
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.activation = activation
        self.task_type = task_type
        self.activation_derivative = activation_derivative

        if task_type == "binary":
            self.output_activation = sigmoid
        elif task_type == "multiclass":
            self.output_activation = softmax
        else:
            self.output_activation = lambda x: x

        # ----- Layer sizes -----
        layer_sizes = (
            [self.input_size]
            + [self.neurons_per_layer] * self.hidden_layers
            + [self.output_size]
        )

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i+1], layer_sizes[i])
            )
            self.biases.append(
                np.random.randn(layer_sizes[i+1], 1)
            )

    def forward(self, embeddings):
        activations = [embeddings]
        weighted_sums = []

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            weighted_sums.append(z)
            a = self.activation(z)
            activations.append(a)

        # Output layer
        z_out = self.weights[-1] @ activations[-1] + self.biases[-1]
        weighted_sums.append(z_out)
        preds = self.output_activation(z_out)

        return preds, activations, weighted_sums

    def backward(self, embeddings, y):
        batch_size = y.shape[1]

        preds, activations, weighted_sums = self.forward(embeddings)

        # ----- Output layer delta -----
        if self.task_type in ["binary", "multiclass"]:
            delta = preds - y
        else:
            delta = preds - y

        gradients_w = [None] * len(self.weights)
        gradients_b = [None] * len(self.biases)

        # ----- Output layer gradients -----
        gradients_w[-1] = (delta @ activations[-2].T) / batch_size
        gradients_b[-1] = np.mean(delta, axis=1, keepdims=True)

        # ----- Hidden layers -----
        for i in range(len(self.weights) - 2, -1, -1):
            delta = (
                self.weights[i + 1].T @ delta
            ) * self.activation_derivative(weighted_sums[i])

            gradients_w[i] = (delta @ activations[i].T) / batch_size
            gradients_b[i] = np.mean(delta, axis=1, keepdims=True)

        # ----- Gradient w.r.t. concatenated embeddings -----
        grad_embeddings = self.weights[0].T @ delta

        return gradients_w, gradients_b, grad_embeddings


    def split_embedding_gradients(self, grad_embeddings):
        grads = {}
        for i, client_id in enumerate(self.client_ids):
            start = i * self.embedding_dim
            end = start + self.embedding_dim
            grads[client_id] = grad_embeddings[start:end]
        return grads

    
    def train(
        self,
        trainloader,      # yields {"y": y}
        clients,
        max_epochs=50,
        convergence_threshold=1e-6,
    ):
        prev_loss = float("inf")

        for epoch in range(max_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in trainloader:
                y = batch["y"]  # (output_size, batch_size)

                # 1️⃣ Client forward passes
                client_activations = []
                for client in clients:
                    h_i = client.forward()  # client uses its own batch internally
                    client_activations.append(h_i)

                # 2️⃣ Concatenate activations
                h = np.vstack(client_activations)

                # 3️⃣ Server forward
                activations, weighted_sums = self.feedforward(h)
                y_hat = activations[-1]

                # 4️⃣ Loss
                loss = self.compute_loss(y_hat, y)
                total_loss += loss
                num_batches += 1

                # 5️⃣ Server backward (w.r.t. h)
                grad_h = self.backward_from_output(y_hat, y)

                # 6️⃣ Split gradients per client
                grad_parts = []
                start = 0
                for client in clients:
                    dim = client.output_dim
                    grad_parts.append(grad_h[start:start + dim, :])
                    start += dim

                # 7️⃣ Send gradients to clients
                for client, g in zip(clients, grad_parts):
                    client.backward(g)

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

            if abs(prev_loss - avg_loss) < convergence_threshold:
                print(f"Converged at epoch {epoch}")
                break

            prev_loss = avg_loss

    def predict(self, testloader):
        all_predictions = []
        all_prediction_probs = []
        all_real_values = []

        total_time = 0.0
        total_samples = 0

        for batch in testloader:
            x = batch["x"]  # (input_size, batch_size)
            y = batch["y"]  # (output_size, batch_size)

            batch_size = x.shape[1]

            # ---- Inference timing ----
            start = time.perf_counter()
            activations, _ = self.feedforward(x)
            end = time.perf_counter()
            total_time += (end - start)
            total_samples += batch_size

            y_hat = activations[-1]

            # ---- Predictions & probabilities ----
            if self.task_type == "binary":
                probs = y_hat.flatten()
                preds = (probs >= 0.5).astype(int)
                real = y.flatten()
                all_prediction_probs.extend(np.vstack([1 - probs, probs]).T)

            elif self.task_type == "multiclass":
                probs = y_hat.T
                preds = np.argmax(probs, axis=1)
                real = np.argmax(y, axis=0)
                all_prediction_probs.extend(probs)

            else:  # regression
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
    
    def load_centralized_labels(
        self,
        batch_size: int = None,
        subset_size: int = -1
    ):
        """
        Load centralized labels for VFL server.
        Server sees LABELS ONLY.
        """

        if batch_size is None:
            raise ValueError("batch_size must be provided for server label loading")

        # ------------------------------------------------------
        # Create dataset + partitioner (same as clients)
        # ------------------------------------------------------
        partitioner = VerticalSizePartitioner(
            partition_sizes=PARTITION_SIZES,
            active_party_columns="target",
            active_party_columns_mode="create_as_last",
        )

        loaded_dataset = load_from_disk(DATASET_DIR)
        partitioner.dataset = loaded_dataset["train"]

        # ------------------------------------------------------
        # Server loads ONLY the active party (labels)
        # Convention: server uses partition_id = -1
        # ------------------------------------------------------
        label_partition = partitioner.load_partition(3)

        print("[Server] label columns:", label_partition.column_names)

        # ---- Safety: ensure only Label is present ----
        assert label_partition.column_names == ["target"]

        # ---- Deterministic train / test split ----
        label_partition = label_partition.train_test_split(
            test_size=0.2,
            seed=SEED,
        )

                # Determine subset
        if 0 < subset_size < len(label_partition["train"]):
            train_labels = label_partition["train"].select(range(subset_size))
            test_labels = label_partition["test"].select(range(subset_size))
        else:
            train_labels = label_partition["train"]
            test_labels = label_partition["test"]

        train_cols = train_labels.column_names
        test_cols = train_labels.column_names

        g = torch.Generator().manual_seed(SEED)

        # Convert once to NumPy arrays (labels only)
        # assuming label column is 'y'
        y_train = np.array(train_labels["target"]).reshape(1, -1).astype(np.float32)  # shape (output_size, N)
        y_test = np.array(test_labels["target"]).reshape(1, -1).astype(np.float32)

        # Optionally, split into batches manually for server-side iteration
        num_samples = y_train.shape[1]
        num_batches = (num_samples + batch_size - 1) // batch_size

        def batch_generator(y_data):
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, num_samples)
                yield {"y": y_data[:, start:end]}

        return batch_generator(y_train), batch_generator(y_test), num_samples
    
    def apply_transforms(self, batch):
        """
        Transform HF batch into NumPy arrays compatible with the custom DNN.
        """

        # 1️⃣ Extract target
        y = np.asarray(batch["target"], dtype=np.int64)  # shape: (batch_size,)

        X = np.stack([batch[col] for col in self.feature_cols], axis=1)
        # X shape: (batch_size, input_size)

        # 3️⃣ Transpose for DNN
        X = X.T  # (input_size, batch_size)
        X = X.astype(np.float32)

        # 4️⃣ Encode targets
        if self.task_type == "binary":
            y = y.astype(np.float32).reshape(1, -1)  # (1, batch_size)

        elif self.task_type == "multiclass":
            num_classes = self.num_classes
            y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
            y_onehot[np.arange(y.shape[0]), y] = 1

            y = y_onehot.T

        else:  # regression
            y = y.reshape(self.output_size, -1)

        return {
            "x": X,
            "y": y,
        }



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)