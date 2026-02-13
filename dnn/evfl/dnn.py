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

INPUT_SIZE = 79
OUTPUT_SIZE = 8
HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 32
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
PARTITION_SIZES = [27, 26, 26]