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

from modules import MultiClassDataset, MultiClassMLP

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
    y = iris.data.targets.to_numpy()
    label2id = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    y = np.vectorize(label2id.get)(y).astype(np.int64)

    # ========================================

    num_classes = np.unique(y).shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args["test"]["size"],
            stratify=y,
            random_state=args["train"]["seed"],
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args["val"]["size"],
        stratify=y_train,
        random_state=args["train"]["seed"],
    )

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_val = standard_scaler.transform(X_val)
    X_test = standard_scaler.transform(X_test)

    train_dataset = MultiClassDataset(X_train, y_train)
    val_dataset = MultiClassDataset(X_val, y_val)
    test_dataset = MultiClassDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["val"]["batch"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)


    model = MultiClassMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=float(args["train"]["lr"]))

    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      num_classes=num_classes,
                      epochs=args["train"]["epochs"],
                      device=device,
                      early_stopping_patience=2)

    model = trainer.fit(train_loader=train_loader, val_loader=val_loader)
    predictions = trainer.predict(test_loader)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    app.run()