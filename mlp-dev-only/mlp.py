from fenn.utils import set_seed
from fenn.trainer import Trainer
from fenn import entrypoint

import pandas as pd
import numpy as np
from pathlib import Path

import joblib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

except ImportError as e:
    raise RuntimeError(
        "Torch is required for this feature. "
        "Install it yourself (GPU/CPU) or use 'pip install smle[torch]'."
    ) from e

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise RuntimeError(
        "ucimlrepo is required for this template. "
        "Install it yourself via 'pip install ucimlrepo'."
    ) from e

class CustomMLP(nn.Module):


    def __init__(self):
        super().__init__()

        self._in_h1 = nn.Linear(11, 18)
        self._h1_h2 = nn.Linear(18, 16)
        self._h2_out = nn.Linear(16, 7)

    def forward(self, x):
        x = F.relu(self._in_h1(x))
        x = F.relu(self._h1_h2(x))
        x = self._h2_out(x)
        return x


class CustomDataset(Dataset):


    def __init__(self, X, y):


        super().__init__()
        self._X = X
        self._y = y


    def __len__(self):
        return len(self._X)


    def __getitem__(self, index):

        x = torch.tensor(self._X[index], dtype=torch.float)
        y = torch.tensor(self._y[index], dtype=torch.long) # Cross Entropy requires long

        return x, y


@entrypoint
def main(args):

    set_seed(args["training"]["seed"])
    export_dir = Path(args["training"]["export_dir"])

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # ========================================
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy()
    y_mapped = y - y.min()
    y = y_mapped.ravel()
    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # ========================================

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=args["training"]["seed"])

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    joblib.dump(standard_scaler, export_dir/ "scaler.joblib")

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=args["training"]["train_batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training"]["test_batch"], shuffle=False)


    device = args["training"]["device"]


    model = CustomMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(args["training"]["learning_rate"]),
                            weight_decay=float(args["training"]["weight_decay"]))


    trainer = Trainer(model, loss_fn, optimizer, device=args["training"]["device"])


    model = trainer.fit(train_loader=train_loader,
                        epochs=args["training"]["epochs"],
                        export_dir=export_dir)

    predictions = []
    grounds = []

    model.eval()
    for data,labels in test_loader:

        data = data.to(device)

        probs = model(data).squeeze()
        preds = torch.argmax(probs, axis=1)

        predictions.extend(preds.detach().cpu().tolist())
        grounds.extend(labels.detach().cpu().tolist())

    print(classification_report(grounds, predictions, zero_division=0))
    print(f"{confusion_matrix(grounds, predictions)}")


if __name__ == "__main__":
    main()