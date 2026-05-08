"""Baselines, XGBoost, and LSTM models for RUL prediction."""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self.mean_)


def train_linear(X_train, y_train):
    m = LinearRegression()
    m.fit(X_train, y_train)
    return m


def train_random_forest(X_train, y_train, seed: int = 42):
    m = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=5,
        random_state=seed, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    return m


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def _xgb_default_params(seed: int = 42):
    """Return sensible defaults — uses GPU if available, CPU otherwise."""
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        # torch not installed OR torch install broken (Windows DLL issues, etc.)
        use_cuda = False
    params = dict(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        seed=seed,
    )
    if use_cuda:
        params["device"] = "cuda"
    else:
        # Use all CPU threads when GPU isn't available.
        import multiprocessing as mp
        params["nthread"] = mp.cpu_count()
    return params


def train_xgb(X_train, y_train, X_val, y_val, params=None,
              num_boost_round: int = 2000, early_stop: int = 100, verbose: int = 100):
    import xgboost as xgb
    params = params or _xgb_default_params()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=verbose,
    )
    return model


def xgb_predict(model, X):
    """Works for both GPU and CPU models — XGBoost auto-routes inference."""
    import xgboost as xgb
    return model.predict(xgb.DMatrix(X))


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
def build_lstm(n_sensors: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
    import torch.nn as nn
    class RULLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_sensors, hidden_size=hidden,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)
    return RULLSTM()


def train_lstm(X_train, y_train, X_val, y_val,
               epochs: int = 100, batch: int = None, lr: float = 1e-3,
               patience: int = 15, ckpt_path: str = "best_lstm.pt",
               hidden: int = 64, num_layers: int = 2, dropout: float = 0.2,
               num_workers: int = 0, grad_clip: float = 1.0,
               normalize_targets: bool = True):
    """
    Train the LSTM.  Key fixes vs. the earlier version:
    - normalize_targets: divide y by R_EARLY (125) so MSE gradients are bounded.
      Predictions are de-normalized before returning.
    - grad_clip:        prevents gradient explosion during early epochs.
    - longer patience:  LSTMs need ~10 epochs to 'wake up' before they beat
      the mean predictor.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = (device == "cuda")

    if batch is None:
        batch = 512 if device == "cuda" else 128

    # Scale targets to [0, 1] so MSE gradients don't explode for large RUL values.
    y_scale = 125.0 if normalize_targets else 1.0
    y_train_s = y_train / y_scale
    y_val_s   = y_val   / y_scale

    model = build_lstm(n_sensors=X_train.shape[2], hidden=hidden,
                       num_layers=num_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          pin_memory=pin, num_workers=num_workers)

    Xv = torch.tensor(X_val, dtype=torch.float32).to(device, non_blocking=pin)
    yv = torch.tensor(y_val_s, dtype=torch.float32).to(device, non_blocking=pin)

    best_val, bad = float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            # Gradient clipping prevents the explosion that causes LSTMs to
            # collapse to a constant prediction in the first few epochs.
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_ds)

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()

        # Report un-scaled RMSE so the number is interpretable during training.
        train_rmse = (tl ** 0.5) * y_scale
        val_rmse   = (vl ** 0.5) * y_scale
        history.append({"epoch": epoch, "train_mse": tl, "val_mse": vl,
                        "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(f"epoch {epoch:3d}  train_rmse={train_rmse:6.2f}  val_rmse={val_rmse:6.2f}")

        # Use a looser tolerance — 0.01 on normalized MSE is ~0.1 RUL cycles
        if vl < best_val - 1e-4:
            best_val = vl
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    # Store the scale on the model so lstm_predict can undo it.
    model._y_scale = y_scale
    return model, pd.DataFrame(history)


def lstm_predict(model, X, batch: int = 512):
    import torch
    device = next(model.parameters()).device
    model.eval()
    y_scale = getattr(model, "_y_scale", 1.0)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) * y_scale
