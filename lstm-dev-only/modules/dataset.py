from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X, y):
        """
        X: torch.Tensor [N, 50, 12] (float32)
        y: torch.Tensor [N] (int64 class indices)
        """
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]