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
    dnn_to_arrays,
    arrays_to_dnn,
    relu
)

class ServerDNN:
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, learning_rate, activation, task_type="multiclass"):
        assert task_type in ["binary", "multiclass", "regression"], \
            f"Unsupported task_type: {task_type}"
        self.fds = None
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

    def forward(self, embeddings):
        z = self.weights[0] @ embeddings + self.biases[0]
        return softmax(z)

    def backward(self, embeddings, y):
        preds = self.forward(embeddings)
        delta = preds - y
        grad_w = delta @ embeddings.T
        grad_b = np.mean(delta, axis=1, keepdims=True)
        grad_embeddings = self.weights[0].T @ delta
        return grad_w, grad_b, grad_embeddings
    
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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)