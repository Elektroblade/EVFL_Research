import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


# =========================
# Global state (loaded once)
# =========================
_DATA_LOADED = False
_X_train = None
_X_test = None
_y_train = None
_y_test = None
_CLIENT_FEATURE_MAP = {}

NUM_CLIENTS = 21
RANDOM_STATE = 42
TEST_SIZE = 0.2
EMBED_DIM = 10


LABEL_COL = "Label"
TARGET_COL = "Target Type"


# =========================
# Load + preprocess
# =========================
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    # Separate labels
    labels = df[LABEL_COL]
    features = df.drop(columns=[LABEL_COL, TARGET_COL])

    # Train/test split (rows must align across clients!)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return X_train, X_test, y_train, y_test


# =========================
# Column → client assignment
# =========================
def build_client_feature_map(feature_columns):
    """
    Assigns feature columns to 21 clients.
    Last client gets remaining 3 features.
    """
    feature_columns = list(feature_columns)
    total_features = len(feature_columns)

    base = total_features // NUM_CLIENTS
    remainder = total_features % NUM_CLIENTS

    feature_map = {}
    idx = 0

    for cid in range(NUM_CLIENTS):
        take = base + (1 if cid < remainder else 0)
        feature_map[str(cid)] = feature_columns[idx : idx + take]
        idx += take

    return feature_map


# =========================
# Public loader (called once)
# =========================
def load_data(csv_path: str = "insdn_data.csv"):
    global _DATA_LOADED, _X_train, _X_test, _y_train, _y_test, _CLIENT_FEATURE_MAP

    if _DATA_LOADED:
        return

    _X_train, _X_test, _y_train, _y_test = load_and_preprocess(csv_path)

    _CLIENT_FEATURE_MAP = build_client_feature_map(_X_train.columns)

    _DATA_LOADED = True


# =========================
# Client-side access
# =========================
def load_client_partition(client_cid: str, train: bool = True):
    """
    Returns ONLY the feature columns assigned to this client.
    """
    if not _DATA_LOADED:
        raise RuntimeError("load_data() must be called before loading partitions")

    cols = _CLIENT_FEATURE_MAP[client_cid]

    if train:
        return _X_train[cols]
    else:
        return _X_test[cols]


# =========================
# Server-side access
# =========================
def load_server_labels(train: bool = True):
    if not _DATA_LOADED:
        raise RuntimeError("load_data() must be called before loading labels")

    return _y_train if train else _y_test


# =========================
# Metadata
# =========================
def get_num_clients():
    return NUM_CLIENTS


def get_num_classes():
    return int(_y_train.nunique())


class ClientModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x.float()              # (batch_size, input_dim)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_CLIENTS * EMBED_DIM, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.fc1(embeddings)
        x = torch.relu(x)
        x = self.bn(x)
        x = self.fc2(x)
        return self.sigmoid(x)



def evaluate_head_model(
    head: ServerModel, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute accuracy of head."""
    head.eval()
    with torch.no_grad():
        correct = 0
        # Re-compute embeddings for accuracy (detached from grad)
        embeddings_eval = embeddings.detach()
        output = head(embeddings_eval)
        predicted = (output > 0.5).float()
        correct += (predicted == labels).sum().item()
        accuracy = correct / len(labels) * 100

    return accuracy

def get_client_input_dim(client_cid: str) -> int:
    if not _DATA_LOADED:
        raise RuntimeError("load_data() must be called first")
    return len(_CLIENT_FEATURE_MAP[client_cid])
