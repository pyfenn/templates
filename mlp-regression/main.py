from fenn import Fenn
from fenn.utils import set_seed
from fenn.nn import RegressionTrainer as Trainer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import os
import kagglehub


from modules.dataset import RegressionDataset
from modules.model import RegressionMLP

app = Fenn()

@app.entrypoint
def main(args):

    device = args["general"]["device"] if torch.cuda.is_available() else "cpu"
    set_seed(args["train"]["seed"]) # set seed for reproducibility

    # ========================================
    # TODO: REPLACE WITH YOUR ACTUAL DATA
    # ========================================
    path = kagglehub.dataset_download("harishkumardatalab/housing-price-prediction", output_dir=args["general"]["dataset_dir"])
    filename = "Housing.csv"
    df = pd.read_csv(os.path.join(path,filename))

    obj_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            obj_cols.append(col)

    df = pd.get_dummies(df, columns=obj_cols).astype(int) # one-hot encoding of object features

    X = df.drop(columns=["price"]).values
    y = df["price"].values

    # ========================================

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=args["test"]["size"],
                                                        random_state=args["train"]["seed"])

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size=args["val"]["size"],
                                                    random_state=args["train"]["seed"])

    # Normalize features using training statistics only
    X_standard_scaler = StandardScaler()
    X_train = X_standard_scaler.fit_transform(X_train)
    X_val = X_standard_scaler.transform(X_val)
    X_test = X_standard_scaler.transform(X_test)

    # Normalize targets using training statistics only
    y_standard_scaler = StandardScaler()
    y_train = y_standard_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val = y_standard_scaler.transform(y_val.reshape(-1, 1))
    y_test = y_standard_scaler.transform(y_test.reshape(-1, 1))

    y_train = y_train.squeeze() # [N, 1] -> [N,]
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    test_dataset = RegressionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    # Validation set (used during training for early stopping)
    val_loader = DataLoader(val_dataset, batch_size=args["test"]["batch"], shuffle=False)
    # Test set (for final evaluation, never seen during training)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)

    model = RegressionMLP()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=float(args["train"]["lr"]))

    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optim=optimizer,
                      device=device)

    trainer.fit(train_loader=train_loader, epochs=args["train"]["epochs"],
                       val_loader=val_loader)

    predictions = trainer.predict(test_loader)

    print(f"R2: {r2_score(y_true=y_test, y_pred=predictions):.2f}")
    print(f"MSE: {mean_squared_error(y_true=y_test, y_pred=predictions):.2f}")
    print(f"MAE: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}")

if __name__ == "__main__":
    app.run()