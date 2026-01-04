from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainer import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from ucimlrepo import fetch_ucirepo

from modules import MultiClassDataset, CustomMLP

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

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=args["test"]["size"],
                                                        stratify=y,
                                                        random_state=args["train"]["seed"])

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    train_dataset = MultiClassDataset(X_train, y_train)
    test_dataset = MultiClassDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)

    model = CustomMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=float(args["train"]["lr"]))

    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      epochs=args["train"]["epochs"],
                      device=device)

    model = trainer.fit(train_loader=train_loader)

    predictions = []
    grounds = []

    model.eval()

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)

            probs = model(data).squeeze()
            preds = torch.argmax(probs, axis=1)

            predictions.extend(preds.detach().cpu().tolist())
            grounds.extend(labels.detach().cpu().tolist())

    print(f"Accuracy: {accuracy_score(grounds, predictions):.4f}")

if __name__ == "__main__":
    app.run()