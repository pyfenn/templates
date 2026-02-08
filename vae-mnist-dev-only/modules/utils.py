from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> "AverageMeter":
        return AverageMeter(self.total + value * n, self.count + n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
