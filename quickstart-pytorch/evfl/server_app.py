
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr.server
import flwr.common
from flwr.common import Scalar, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

import vertical_fl.task # Import the entire task module

# Define the server-side model which takes concatenated embeddings from clients
class ServerNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vertical_fl.task.load_data() # Ensure data is loaded when strategy is initialized
        self.num_classes = vertical_fl.task.get_num_classes()
        self.num_clients = vertical_fl.task.get_num_clients()
        self.embedding_dim = 10 # Assuming each client sends 10-dim embedding from ClientNet
        self.server_model = ServerNet(self.num_clients * self.embedding_dim, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.server_model.to(self.device)

        # Load server labels for evaluation
        self.train_labels = vertical_fl.task.load_server_labels(train=True)
        self.test_labels = vertical_fl.task.load_server_labels(train=False)

        # Convert labels to PyTorch DataLoader
        self.train_label_loader = DataLoader(TensorDataset(torch.tensor(self.train_labels.values, dtype=torch.long)), batch_size=32, shuffle=True)
        self.test_label_loader = DataLoader(TensorDataset(torch.tensor(self.test_labels.values, dtype=torch.long)), batch_size=32)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters, metrics_aggregated

    def evaluate(self, server_round: int, parameters: flwr.common.NDArrays) -> tuple[float, dict[str, Scalar]]:
        self.server_model.train() # Train the server model using the labels
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (labels_batch,) in enumerate(self.train_label_loader):
            dummy_embeddings = torch.randn(labels_batch.size(0), self.num_clients * self.embedding_dim).to(self.device)
            labels_batch = labels_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.server_model(dummy_embeddings)
            loss = self.loss_fn(outputs, labels_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        avg_loss = total_loss / len(self.train_label_loader)
        accuracy = correct / total

        self.server_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (labels_batch,) in enumerate(self.test_label_loader):
                dummy_embeddings = torch.randn(labels_batch.size(0), self.num_clients * self.embedding_dim).to(self.device)
                labels_batch = labels_batch.to(self.device)
                outputs = self.server_model(dummy_embeddings)
                loss = self.loss_fn(outputs, labels_batch)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels_batch.size(0)
                test_correct += (predicted == labels_batch).sum().item()

        avg_test_loss = test_loss / len(self.test_label_loader)
        test_accuracy = test_correct / test_total

        return float(avg_test_loss), {"accuracy": float(test_accuracy)}

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = 1
    if hasattr(context, 'num_rounds'):
        num_rounds = context.num_rounds

    num_server_rounds_eval = num_rounds

    strategy = CustomStrategy(
        min_fit_clients=vertical_fl.task.get_num_clients(),
        min_evaluate_clients=vertical_fl.task.get_num_clients(),
    )

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)
