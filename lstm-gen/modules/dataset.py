import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Character-level dataset. Builds a vocabulary from the raw text and
    produces (input_seq, target_seq) pairs where target is input shifted
    right by one position — the standard next-character prediction task.
    """

    def __init__(self, text: str, seq_len: int):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        # Sliding window: X[i] = data[i:i+seq_len], y[i] = data[i+1:i+seq_len+1]
        n = len(data) - seq_len
        self.X = torch.stack([data[i:i + seq_len]     for i in range(n)])
        self.y = torch.stack([data[i + 1:i + seq_len + 1] for i in range(n)])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
