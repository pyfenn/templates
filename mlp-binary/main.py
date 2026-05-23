from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn import ClassificationTrainer as Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from datasets import load_dataset

from modules import BinaryDataset, BinaryMLP

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    # ========================================
    # Load Iris dataset using Hugging Face datasets
    # ========================================
    iris = load_dataset("scikit-learn/iris")
    data = iris["train"]

    X = np.array([
        [
            row["SepalLengthCm"],
            row["SepalWidthCm"],
            row["PetalLengthCm"],
            row["PetalWidthCm"],
        ]
        for row in data
    ], dtype=np.float32)

    y_raw = np.array(data["Species"])

    # setosa = 1, others = 0
    y = (y_raw == "Iris-setosa").astype(np.int64)

    num_classes = np.unique(y).shape[0]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=args["train"]["seed"],
    )

    # Second split: 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=args["train"]["seed"],
    )

    # Normalize using only training statistics
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_val = standard_scaler.transform(X_val)
    X_test = standard_scaler.transform(X_test)

    train_dataset = BinaryDataset(X_train, y_train)
    val_dataset = BinaryDataset(X_val, y_val)
    test_dataset = BinaryDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["train"]["batch"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test"]["batch"],
        shuffle=False,
    )

    model = BinaryMLP()

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args["train"]["lr"]),
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        num_classes=num_classes,
        device=device,
        early_stopping_patience=5,
    )

    trainer.fit(
        train_loader=train_loader,
        epochs=args["train"]["epochs"],
        val_loader=val_loader,
    )

    predictions = trainer.predict(test_loader)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")


if __name__ == "__main__":
    app.run()