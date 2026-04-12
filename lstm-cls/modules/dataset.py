import numpy as np
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    """Wraps pre-split X / y arrays as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)            # (N, T, F) float32
        self.y = torch.from_numpy(y).long()     # (N,)   int64

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
