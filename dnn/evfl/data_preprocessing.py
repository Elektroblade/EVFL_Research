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

INPUT_SIZE = 78
OUTPUT_SIZE = 3
HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 64
TASK_TYPE = "multiclass"
SEED = 42
BATCH_SIZE = 32
DROP_COLS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp"
]
DATASET_DIR = "datasets/insdn_hf_dataset"
target_column: str = "Label"

def main():
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

    num_classes = insdn_data_df[target_column].nunique()

    insdn_data_df["target"] = insdn_data_df[target_column]

    insdn_data_df = insdn_data_df.drop(columns=DROP_COLS, errors="ignore")

    columns_without_labels = insdn_data_df.drop(columns=['Label', 'Target Type']).columns
    print("Columns in the dataset (excluding 'Label' and 'Target Type'):")
    print(list(columns_without_labels))

    hf_dataset = Dataset.from_pandas(insdn_data_df)
    hf_dataset = hf_dataset.remove_columns(
        [c for c in ["Label", "Target Type"]]
    )
    hf_dataset = hf_dataset.class_encode_column("target")
    dataset_dict = DatasetDict({
        "train": hf_dataset
    })

    #dataset_dict["train"].info.metadata = {"min_partition_size": 2}  # or whatever number makes sense

    dataset_dict.save_to_disk(DATASET_DIR)

    ds = load_from_disk(DATASET_DIR)
    print(ds["train"].column_names)
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