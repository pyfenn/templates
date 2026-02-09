from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn.trainers import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ucimlrepo import fetch_ucirepo 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

from modules.dataset import SeqDataset
from modules.model import LSTMClassifier
from modules.utils import build_X_y_lstm
app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"]) # set seed for reproducibility

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # ========================================
    
    # fetch dataset 
    online_retail = fetch_ucirepo(id=352) 
    
    # data (as pandas dataframes) 
    df = online_retail.data.features
    X, y = build_X_y_lstm(df)
    print(X)
    return
    # ========================================

    train_dataset = TrajDataset(X_train, y_train)
    test_dataset = TrajDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)

    model = LSTMClassifier(input_dim=12, num_classes=4, bidirectional=True, pooling="last")
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

            probs = model(data)
            preds = torch.argmax(probs, axis=1)

            predictions.extend(preds.detach().cpu().tolist())
            grounds.extend(labels.detach().cpu().tolist())

    print(f"Accuracy: {accuracy_score(grounds, predictions):.4f}")

if __name__ == "__main__":
    app.run()