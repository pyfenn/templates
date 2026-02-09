import numpy as np
import pandas as pd

# df: Online Retail con colonne tipiche:
# ['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']

def build_X_y_lstm(df: pd.DataFrame, freq="W", seq_len=12, horizon=1):
    df = df.copy()

    # pulizia minima (per avere sequenze sensate)
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # spesso conviene togliere resi/cancellazioni (quantità <= 0)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # feature base per riga
    df["amount"] = df["Quantity"] * df["UnitPrice"]

    # aggregazione temporale per cliente
    df["period"] = df["InvoiceDate"].dt.to_period(freq).dt.start_time

    agg = (
        df.groupby(["CustomerID", "period"])
          .agg(
              n_invoices=("InvoiceNo", "nunique"),
              n_items=("Quantity", "sum"),
              revenue=("amount", "sum"),
              n_unique_products=("StockCode", "nunique"),
          )
          .reset_index()
          .sort_values(["CustomerID", "period"])
    )

    # presenza acquisto nel periodo (serve per la label)
    agg["active"] = 1

    # per avere anche i periodi "vuoti" (senza acquisti) nella timeline del cliente
    X_list, y_list = [], []

    feature_cols = ["n_invoices", "n_items", "revenue", "n_unique_products", "active"]

    for cid, g in agg.groupby("CustomerID"):
        g = g.sort_values("period")

        # reindex su timeline completa
        full_idx = pd.date_range(g["period"].min(), g["period"].max(), freq=freq)
        gg = (
            g.set_index("period")[feature_cols]
             .reindex(full_idx)
             .fillna(0.0)
        )

        # label: acquisto nel periodo futuro (horizon)
        future_active = (gg["active"].shift(-horizon) > 0).astype(int)

        # costruiamo esempi (finestre) per LSTM
        # X[t] = seq_len periodi che finiscono in t-1, y[t] = future_active in t-1 (o t)
        for end in range(seq_len, len(gg) - horizon + 1):
            x_win = gg.iloc[end - seq_len:end].to_numpy(dtype=np.float32)
            y_val = future_active.iloc[end - 1]  # label riferita all'ultimo step della finestra
            X_list.append(x_win)
            y_list.append(int(y_val))

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y, feature_cols