import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self._texts = list(texts)
        self._labels = list(labels)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        enc = self._tokenizer(
            self._texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self._labels[idx], dtype=torch.long),
        }
