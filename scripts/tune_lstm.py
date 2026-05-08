"""
Small hyperparameter grid for the LSTM.

LSTMs are slow to train, so we test only 4 hand-picked configs.
Reads the raw windows from .npz files written by extract_features.py.
GPU is auto-used inside train_lstm().

Usage:
    python tune_lstm.py --subset FD001
    python tune_lstm.py --subset FD002 --epochs 80
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config
from evaluate import rmse, nasa_score
from models import train_lstm, lstm_predict


def load_windows(subset: str):
    tr  = np.load(f"{config.PROCESSED_DIR}/{subset}_windows_train.npz")
    val = np.load(f"{config.PROCESSED_DIR}/{subset}_windows_val.npz")
    return (tr["X"].astype(np.float32),  tr["y"].astype(np.float32),
            val["X"].astype(np.float32), val["y"].astype(np.float32))


# Hand-picked configs to test. Keep this small — LSTMs are expensive.
LSTM_CONFIGS = [
    {"name": "small",       "hidden":  32, "num_layers": 1, "dropout": 0.1, "lr": 1e-3},
    {"name": "default",     "hidden":  64, "num_layers": 2, "dropout": 0.2, "lr": 1e-3},
    {"name": "wider",       "hidden": 128, "num_layers": 2, "dropout": 0.3, "lr": 1e-3},
    {"name": "deeper_slow", "hidden":  64, "num_layers": 3, "dropout": 0.3, "lr": 5e-4},
]


def run_trial(X_tr, y_tr, X_val, y_val, cfg, subset, epochs, patience):
    """Train one LSTM config, return best val metrics."""
    ckpt = f"{config.MODELS_DIR}/lstm_tuning_{subset}_{cfg['name']}.pt"
    t0 = time.time()
    model, hist = train_lstm(
        X_tr, y_tr, X_val, y_val,
        epochs=epochs, batch=config.LSTM_BATCH,
        lr=cfg["lr"], patience=patience, ckpt_path=ckpt,
        hidden=cfg["hidden"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    elapsed = time.time() - t0

    # model has best-epoch weights loaded
    y_pred_val = lstm_predict(model, X_val)
    best_epoch = int(hist["val_mse"].idxmin())
    return {
        "val_rmse": rmse(y_val, y_pred_val),
        "val_nasa": nasa_score(y_val, y_pred_val),
        "best_epoch": best_epoch,
        "epochs_run": int(hist["epoch"].iloc[-1]) + 1,
        "time_s": round(elapsed, 1),
    }


def tune(subset: str, epochs: int, patience: int):
    X_tr, y_tr, X_val, y_val = load_windows(subset)
    print(f"Shapes: X_tr={X_tr.shape}, X_val={X_val.shape}")

    results = []
    for cfg in LSTM_CONFIGS:
        print(f"\n=== Training config: {cfg['name']} "
              f"(hidden={cfg['hidden']}, layers={cfg['num_layers']}, "
              f"dropout={cfg['dropout']}, lr={cfg['lr']}) ===")
        m = run_trial(X_tr, y_tr, X_val, y_val, cfg, subset, epochs, patience)
        results.append({**cfg, **m})
        print(f"val_rmse={m['val_rmse']:.3f}  val_nasa={m['val_nasa']:.0f}  "
              f"best_epoch={m['best_epoch']}  time={m['time_s']}s")

    df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    print(f"\n=== Ranked by val_rmse ===")
    print(df.to_string(index=False))

    best_cfg = df.iloc[0].to_dict()
    out_json = f"{config.MODELS_DIR}/lstm_{subset}_best_params.json"
    with open(out_json, "w") as f:
        keep = ["name", "hidden", "num_layers", "dropout", "lr",
                "val_rmse", "val_nasa", "best_epoch"]
        json.dump({k: best_cfg[k] for k in keep}, f, indent=2,
                  default=lambda x: x.item() if hasattr(x, "item") else x)
    print(f"\nWrote {out_json}")

    out_csv = f"{config.MODELS_DIR}/lstm_{subset}_tuning_log.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    return best_cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    config.print_hardware()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tune(args.subset, args.epochs, args.patience)