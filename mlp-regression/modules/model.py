import torch.nn as nn
import torch.nn.functional as F

class RegressionMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self._in_h1 = nn.Linear(20, 32)
        self._h1_h2 = nn.Linear(32, 32)
        self._h2_out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self._in_h1(x))
        x = F.relu(self._h1_h2(x))
        x = self._h2_out(x)
        return x