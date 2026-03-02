from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from ucimlrepo import fetch_ucirepo

from modules import BinaryDataset, BinaryMLP

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"]) # set seed for reproducibility

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # ========================================
    iris = fetch_ucirepo(name="iris")
    X = iris.data.features.to_numpy()
    y = iris.data.targets.to_numpy().ravel()

    # setosa = 1, others = 0
    y = (y == "Iris-setosa").astype(np.int64)
    
    # ========================================
    
    num_classes = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=args["test"]["size"],
                                                        stratify=y,
                                                        random_state=args["train"]["seed"])

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    train_dataset = BinaryDataset(X_train, y_train)
    test_dataset = BinaryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)

    model = BinaryMLP()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=float(args["train"]["lr"]))

    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      num_classes=num_classes,
                      epochs=args["train"]["epochs"],
                      device=device)

    model = trainer.fit(train_loader=train_loader)
    predictions = trainer.predict(test_loader)

    model.eval()

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    app.run()