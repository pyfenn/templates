import torch
from torch.utils.data import Dataset

class MultiClassDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, index):
        x = torch.tensor(self._X[index], dtype=torch.float)
        y = torch.tensor(self._y[index], dtype=torch.long) # cross entropy requires long
        y = y.squeeze(0) # [B, 1] -> [B] - for cross entropy
        return x, y