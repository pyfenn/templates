from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import ClassificationTrainer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from modules import SeqDataset, LSTMClassifier

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"])

    # ========================================
    # Data — WISDM Human Activity Recognition
    # 17,166 accelerometer sequences (3-axis, 100 timesteps each).
    # 6 classes: Walking, Jogging, Stairs, Sitting, Standing, Lying Down.
    # Files are downloaded from HuggingFace Hub on first run and cached locally.
    # The dataset uses a custom loading script so we fetch the raw .npy files
    # directly via huggingface_hub instead of load_dataset().
    # ========================================
    X_path = hf_hub_download(repo_id="monster-monash/WISDM", filename="WISDM_X.npy", repo_type="dataset")
    y_path = hf_hub_download(repo_id="monster-monash/WISDM", filename="WISDM_y.npy", repo_type="dataset")

    # X: (17166, 3, 100) channels-first → transpose to (17166, 100, 3) for LSTM.
    X = np.load(X_path).transpose(0, 2, 1).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args["test"]["size"],
        stratify=y,
        random_state=args["train"]["seed"],
    )
    # ========================================

    train_dataset = SeqDataset(X_train, y_train)
    test_dataset  = SeqDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args["test"]["batch"],  shuffle=False)

    model = LSTMClassifier(
        input_dim=args["data"]["features"],
        hidden_dim=args["model"]["hidden_dim"],
        num_layers=args["model"]["num_layers"],
        num_classes=args["model"]["num_classes"],
        dropout=args["model"]["dropout"],
        bidirectional=args["model"]["bidirectional"],
        pooling=args["model"]["pooling"],
    )

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args["train"]["lr"]))

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optim=optimizer,
        num_classes=args["model"]["num_classes"],
        device=device,
    )

    trainer.fit(train_loader=train_loader, epochs=args["train"]["epochs"])
    predictions = trainer.predict(test_loader)

    print(f"Test Accuracy: {accuracy_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    app.run()
