import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, VerticalSizePartitioner
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datasets import Dataset
from datasets import DatasetDict
import time
import random
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_from_disk
from logging import INFO
from flwr.common import log

INPUT_SIZE = 79
OUTPUT_SIZE = 8
HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 32
TASK_TYPE = "multiclass" #"multiclass"
SEED = 42
BATCH_SIZE = 32
DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
]
DATASET_DIR = "datasets/insdn_hf_dataset" # insdn_hf_dataset
TARGET_COLUMN = "target"
FEATURE_COLUMNS = ['Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
                   'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 
                   'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 
                   'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 
                   'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 
                   'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 
                   'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 
                   'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 
                   'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 
                   'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 
                   'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 
                   'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 
                   'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
PARTITION_SIZES = [26,26,25]
DATASET_NAME = "insdn"
MODEL_FAMILY = "DNN"

"""
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Fare",
    "Siblings/Spouses Aboard",
    'embarked_Q', 'embarked_S',
    "Parents/Children Aboard",
    "Pclass",
]
"""


def load_and_preprocess(
    dataframe: pd.DataFrame,
):
    """Preprocess a subset of the titanic-survival dataset columns into a purely
    numerical numpy array suitable for model training."""

    # Make a copy to avoid modifying the original
    X_df = dataframe.copy()

    # Identify which columns are present
    available_cols = set(X_df.columns)

    # ----------------------------------------------------------------------
    # FEATURE ENGINEERING ON NAME (if present)
    # ----------------------------------------------------------------------
    if "Name" in available_cols:
        X_df["Title"] = X_df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        X_df["NameLength"] = X_df["Name"].str.len()
        X_df = X_df.drop(columns=["Name"])

    # ----------------------------------------------------------------------
    # IDENTIFY NUMERIC + CATEGORICAL COLUMNS
    # ----------------------------------------------------------------------
    categorical_cols = []
    if "Sex" in X_df.columns:
        categorical_cols.append("Sex")
    if "Title" in X_df.columns:
        categorical_cols.append("Title")
    if "Pclass" in X_df.columns:
        categorical_cols.append("Pclass")

    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    # ----------------------------------------------------------------------
    # HANDLE MISSING VALUES
    # ----------------------------------------------------------------------
    if numeric_cols:
        X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].median())

    # ----------------------------------------------------------------------
    # PREPROCESSOR (TRANSFORM TO PURE NUMERIC)
    # ----------------------------------------------------------------------
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # ----------------------------------------------------------------------
    # FIT TRANSFORMER & CONVERT TO NUMPY
    # ----------------------------------------------------------------------
    X_full = preprocessor.fit_transform(X_df)

    # Ensure output is always a dense numpy array
    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()

    return X_full.astype(np.float32)


partitioner = None  # Cache FederatedDataset


def load_data(partition_id: int, feature_splits: list[int], subset_size: int):
    """..."""

    global partitioner
    if partitioner is None:
        partitioner = VerticalSizePartitioner(
            partition_sizes=feature_splits,
            active_party_columns="target",
            active_party_columns_mode="create_as_last",
        )

        loaded_dataset = load_from_disk(DATASET_DIR)
        partitioner.dataset = loaded_dataset["train"]
        log(INFO, loaded_dataset["train"].column_names)
        log(INFO, len(loaded_dataset["train"].column_names))

    # Load partition
    partition = partitioner.load_partition(partition_id)
    
    partition = partition.train_test_split(
        test_size=0.2,
        seed=SEED,
    )

    if 0 < subset_size < len(partition["train"]):
        train_dataset = partition["train"].select(range(subset_size))
        test_dataset = partition["test"].select(range(subset_size))
    else:
        train_dataset = partition["train"]
        test_dataset = partition["test"]

    # Process partition
    return load_and_preprocess(dataframe=train_dataset.to_pandas()), load_and_preprocess(dataframe=test_dataset.to_pandas())


class ClientModel(nn.Module):
    def __init__(self, input_size, out_feat_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)          # extra hidden layer
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, out_feat_dim) # output layer

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ServerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ServerModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)   # extra hidden
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        if self.num_classes <= 2:
            self.fc = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc = nn.Linear(64, num_classes)  # multiclass logits

    def forward(self, x):
        if (self.num_classes <= 2):
            x = nn.functional.relu(self.hidden1(x))
            x = self.bn1(x)
            x = nn.functional.relu(self.hidden2(x))
            x = self.bn2(x)
            x = self.fc(x)
            return self.sigmoid(x)
        else:
            x = nn.functional.relu(self.hidden1(x))
            x = self.bn1(x)
            x = nn.functional.relu(self.hidden2(x))
            x = self.bn2(x)
            x = self.fc(x)
            return x
        
def evaluate_head_model(
    head: ServerModel, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute accuracy of head."""
    if TASK_TYPE == "binary":
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
    else:
        head.eval()
        with torch.no_grad():
            logits = head(embeddings)
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels) * 100
        return accuracy