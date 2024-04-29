import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def load_data(file_path):
    df = pl.read_parquet(file_path)
    tensors = [torch.tensor(df[col].to_numpy().reshape(-1, 100), dtype=torch.float32) for col in df.columns]
    return tensors

def train_dataset():
    train_tensors = "data/train.parquet"
    return TensorDataset(*load_data(train_tensors))

def val_dataset():
    val_tensors = "data/val.parquet"
    return TensorDataset(*load_data(val_tensors))