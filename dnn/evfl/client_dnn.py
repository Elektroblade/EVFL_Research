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

class ClientDNN:
    def __init__(self, input_size, hidden_layers, neurons_per_layer, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = []
        self.biases = []
        self.has_dataset = False
        self.feature_cols=None

        layer_sizes = [input_size] + [neurons_per_layer] * hidden_layers

        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i+1], layer_sizes[i])
            )
            self.biases.append(
                np.random.randn(layer_sizes[i+1], 1)
            )

    def forward(self, x):
        activations = x
        for w, b in zip(self.weights, self.biases):
            activations = self.activation(w @ activations + b)
        return activations

    def backward(self, x, grad_from_server):
        """
        Backward pass for ClientDNN in Vertical Federated Learning.

        Args:
            x: client feature matrix, shape (d_client, B)
            grad_from_server: gradient w.r.t. client embedding, shape (h_dim, B)

        Returns:
            gradients_w: gradients for client weights
            gradients_b: gradients for client biases
        """

        batch_size = x.shape[1]

        # ---- Feedforward (client-side only) ----
        activations = [x]
        weighted_sums = []

        num_layers = len(self.weights)

        for i in range(num_layers):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            weighted_sums.append(z)

            # Client always uses hidden activation (no output layer)
            a = self.activation(z)
            activations.append(a)

        # ---- Backward pass starts from server gradient ----
        delta = grad_from_server * self.activation_derivative(weighted_sums[-1])

        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        gradients_w[-1] = (delta @ activations[-2].T) / batch_size
        gradients_b[-1] = np.mean(delta, axis=1, keepdims=True)

        # ---- Propagate through remaining client layers ----
        for i in range(num_layers - 2, -1, -1):
            delta = (self.weights[i + 1].T @ delta) * self.activation_derivative(weighted_sums[i])
            gradients_w[i] = (delta @ activations[i].T) / batch_size
            gradients_b[i] = np.mean(delta, axis=1, keepdims=True)

        # ---- Safety checks ----
        for i, (w, b, dw, db) in enumerate(zip(self.weights, self.biases, gradients_w, gradients_b)):
            assert dw.shape == w.shape, f"Layer {i}: dw {dw.shape} vs w {w.shape}"
            assert db.shape == b.shape, f"Layer {i}: db {db.shape} vs b {b.shape}"

        return gradients_w, gradients_b
    
    def load_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ):
        """
        Load vertically-partitioned NIDS data for a VFL client.
        Client sees FEATURES ONLY.
        """

        if self.has_dataset == False:
            self.partitioner = VerticalSizePartitioner(
                partition_sizes=PARTITION_SIZES,
                active_party_columns="target",          # label exists
                active_party_columns_mode="create_as_last",
            )

            loaded_dataset = load_from_disk(DATASET_DIR)
            self.partitioner.dataset = loaded_dataset["train"]
            self.has_dataset = True

        # ------------------------------------------------------
        # Load THIS CLIENT'S vertical slice (features only)
        # ------------------------------------------------------
        partition = self.partitioner.load_partition(partition_id)

        #print(f"[Client {partition_id}] columns:", partition.column_names)

        # ---- Drop label if present (safety) ----
        if "target" in partition.column_names:
            partition = partition.remove_columns(["target"])

        self.feature_cols = [
            c for c in partition.column_names
        ]

        # ---- Train / test split (must be deterministic) ----
        partition = partition.train_test_split(
            test_size=0.2,
            seed=SEED,
        )

        # ---- Apply preprocessing (normalization, etc.) ----
        #partition = partition.with_transform(self.apply_transforms_no_labels)

        g = torch.Generator().manual_seed(SEED)

        train_dataset = partition["train"]
        test_dataset = partition["test"]

        train_cols = train_dataset.column_names
        test_cols = test_dataset.column_names

        # Convert once to NumPy arrays (features only)
        X_train = np.stack(
            [
                np.array(train_dataset[col])
                for col in train_cols
            ],
            axis=1
        ).astype(np.float32)
        X_test = np.stack(
            [
                np.array(test_dataset[col])
                for col in test_cols
            ],
            axis=1
        ).astype(np.float32)

        return X_train, X_test

    
    def apply_transforms_no_labels(self, batch):
        """
        Client-side VFL transform: FEATURES ONLY
        """

        X = np.stack([batch[col] for col in self.feature_cols], axis=1)
        X = X.T.astype(np.float32)  # (input_size, batch_size)

        return {
            "x": X
        }
    
    def predict(self, testloader):
        """
        Run inference on the client features (ClientDNN).

        Returns:
            activations: list of np.ndarray, each of shape (neurons_last_layer, batch_size)
            real_values: np.ndarray of true labels (only if available)
            avg_inference_time_ms: float
        """
        all_activations = []
        all_real_values = []

        total_time = 0.0
        total_samples = 0

        for batch in testloader:
            x = batch["x"]  # (input_size, batch_size)
            y = batch.get("y")  # may be None for some clients

            batch_size = x.shape[1]

            # ---- Inference timing ----
            start = time.perf_counter()
            activations, _ = self.forward(x)  # use client's forward method
            end = time.perf_counter()

            total_time += (end - start)
            total_samples += batch_size

            all_activations.append(activations[-1])  # only last hidden layer

            if y is not None:
                # reshape for consistency
                all_real_values.extend(y.flatten())

        avg_inference_time_ms = (total_time / total_samples) * 1000.0

        return (
            all_activations,                # list of client activations per batch
            np.array(all_real_values) if all_real_values else None,
            avg_inference_time_ms
        )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

