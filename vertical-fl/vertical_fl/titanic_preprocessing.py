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
from datasets import load_from_disk
import seaborn as sns

INPUT_SIZE = 78
OUTPUT_SIZE = 3
HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 64
TASK_TYPE = "binary"
SEED = 42
BATCH_SIZE = 32
DROP_COLS = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]
DATASET_DIR = "datasets/titanic"
target_column: str = "survived"

def normalize_features(df, target_col):
    df = df.copy()

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove target column if present
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Standardize: (x - mean) / std
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std > 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0.0  # constant column

    return df

def main():    
    insdn_data_df = sns.load_dataset("titanic")
    insdn_data_df = normalize_columns(insdn_data_df)

    insdn_data_df = insdn_data_df[[
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]]

    # Fill missing values
    insdn_data_df["age"] = insdn_data_df["age"].fillna(insdn_data_df["age"].median())
    insdn_data_df["fare"] = insdn_data_df["fare"].fillna(insdn_data_df["fare"].median())
    insdn_data_df["embarked"] = insdn_data_df["embarked"].fillna(
        insdn_data_df["embarked"].mode()[0]
    )

    # Encode categoricals
    insdn_data_df = pd.get_dummies(
        insdn_data_df,
        columns=["sex", "embarked"],
        drop_first=True
    )

    insdn_data_df = normalize_target_column(insdn_data_df, target_column)

    assert target_column in insdn_data_df.columns, \
        f"{target_column} not found in dataset"

    num_classes = insdn_data_df[target_column].nunique()

    insdn_data_df["target"] = insdn_data_df[target_column]

    insdn_data_df = insdn_data_df.drop(columns=DROP_COLS, errors="ignore")

    columns_without_labels = insdn_data_df.drop(columns=[target_column]).columns
    print("Columns in the dataset (excluding 'Label' and 'Target Type'):")
    print(list(columns_without_labels))

    print(f"Labels: {insdn_data_df["target"].unique()}")

    # Normalize full df
    insdn_data_df = normalize_features(insdn_data_df, target_column)

    hf_dataset = Dataset.from_pandas(insdn_data_df)
    hf_dataset = hf_dataset.remove_columns(
        [c for c in [target_column]]
    )
    hf_dataset = hf_dataset.class_encode_column("target")
    dataset_dict = DatasetDict({
        "train": hf_dataset
    })

    num_columns = len(
        [c for c in dataset_dict["train"].column_names if c != "target"]
    )

    print(num_columns)

    dataset_dict.save_to_disk(DATASET_DIR)

    ds = load_from_disk(DATASET_DIR)
    print(ds["train"].column_names)
    print(len(ds["train"].column_names))
    print(ds["train"][0]["target"]) 

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

if __name__ == "__main__":
    main()