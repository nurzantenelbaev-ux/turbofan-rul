"""
Final training + test evaluation for one subset.

Reads tuned hyperparameters from:
    models/xgb_<SUBSET>_best_params.json   (written by tune_xgb.py)
    models/lstm_<SUBSET>_best_params.json  (written by tune_lstm.py)

Reads features from the parquets written by extract_features.py, applies the
feature selection from select_features.py (if available), trains across N seeds,
evaluates once on the test set, and prints a results table.

Usage:
    python train.py --subset FD001
    python train.py --subset FD002 --model xgb
    python train.py --subset FD004 --seeds 7 13 42 101 999
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config
from evaluate import rmse, nasa_score, both_metrics
from models import (
    MeanPredictor, train_linear, train_random_forest,
    train_xgb, xgb_predict,
    train_lstm, lstm_predict,
    _xgb_default_params,
)


# ---------------------------------------------------------------------------
# Feature loading (for XGBoost and classical baselines)
# ---------------------------------------------------------------------------
def load_features(subset: str):
    """Load parquet features and apply feature selection if available."""
    Ftr  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_train.parquet")
    Fval = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_val.parquet")
    Fte  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_test.parquet")

    y_tr, y_val, y_te = Ftr["RUL"].values, Fval["RUL"].values, Fte["RUL"].values
    Xtr  = Ftr.drop(columns=["RUL"])
    Xval = Fval.drop(columns=["RUL"])
    Xte  = Fte.drop(columns=["RUL"])

    sel_path = f"{config.PROCESSED_DIR}/{subset}_selected_features.json"
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            feats = json.load(f)["features"]
        feats = [c for c in feats if c in Xtr.columns]
        Xtr, Xval, Xte = Xtr[feats], Xval[feats], Xte[feats]
        print(f"Using {len(feats)} selected features from {sel_path}")
    else:
        print(f"No feature selection found; using all {Xtr.shape[1]} features.")

    print(f"Shapes: train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}")
    return Xtr, y_tr, Xval, y_val, Xte, y_te


def load_windows(subset: str):
    """Load raw windows for the LSTM."""
    tr  = np.load(f"{config.PROCESSED_DIR}/{subset}_windows_train.npz")
    val = np.load(f"{config.PROCESSED_DIR}/{subset}_windows_val.npz")
    te  = np.load(f"{config.PROCESSED_DIR}/{subset}_windows_test.npz")
    return (tr["X"].astype(np.float32),  tr["y"].astype(np.float32),
            val["X"].astype(np.float32), val["y"].astype(np.float32),
            te["X"].astype(np.float32),  te["y"].astype(np.float32))


def load_best_params(path: str, fallback: dict | None = None):
    """Load tuned hyperparameters; warn and fall back to defaults if missing."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Using defaults.")
        return fallback or {}
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded tuned params from {path}")
    return data


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize_seeds(runs: list):
    """Given a list of {'rmse': ..., 'nasa_score': ...} dicts, return mean/std."""
    rmses = [r["rmse"] for r in runs]
    nasas = [r["nasa_score"] for r in runs]
    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std":  float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "nasa_mean": float(np.mean(nasas)),
        "nasa_std":  float(np.std(nasas, ddof=1)) if len(nasas) > 1 else 0.0,
        "n_seeds":   len(runs),
    }


def print_results_table(results: dict, subset: str):
    """Pretty-print the final comparison table."""
    rows = []
    for name, entry in results.items():
        if isinstance(entry, dict) and "rmse_mean" in entry:
            # Multi-seed summary
            rows.append({
                "model":      name,
                "test_rmse":  f"{entry['rmse_mean']:.2f} ± {entry['rmse_std']:.2f}",
                "test_nasa":  f"{entry['nasa_mean']:.0f} ± {entry['nasa_std']:.0f}",
                "n_seeds":    entry["n_seeds"],
            })
        elif isinstance(entry, dict) and "rmse" in entry:
            # Single-run baseline (no seeds)
            rows.append({
                "model":      name,
                "test_rmse":  f"{entry['rmse']:.2f}",
                "test_nasa":  f"{entry['nasa_score']:.0f}",
                "n_seeds":    1,
            })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS — {subset}")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(subset: str, model_choice: str, seeds: list):
    config.print_hardware()
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    results = {}

    # -----------------------------------------------------------------------
    # XGBoost + classical baselines (all use tabular features)
    # -----------------------------------------------------------------------
    if model_choice in ("xgb", "all"):
        print(f"\n=== Loading features for {subset} ===")
        Xtr, y_tr, Xval, y_val, Xte, y_te = load_features(subset)

        # --- Classical baselines (single run each, deterministic or seed=42) ---
        print("\n--- Classical baselines ---")
        m_mean = MeanPredictor().fit(Xtr.values, y_tr)
        results["mean"] = both_metrics(y_te, m_mean.predict(Xte.values))
        print(f"mean:   rmse={results['mean']['rmse']:.2f}  "
              f"nasa={results['mean']['nasa_score']:.0f}")

        m_lin = train_linear(Xtr.values, y_tr)
        results["linear"] = both_metrics(y_te, m_lin.predict(Xte.values))
        print(f"linear: rmse={results['linear']['rmse']:.2f}  "
              f"nasa={results['linear']['nasa_score']:.0f}")

        m_rf = train_random_forest(Xtr.values, y_tr, seed=config.SEED)
        results["random_forest"] = both_metrics(y_te, m_rf.predict(Xte.values))
        print(f"rf:     rmse={results['random_forest']['rmse']:.2f}  "
              f"nasa={results['random_forest']['nasa_score']:.0f}")

        # --- XGBoost with tuned params, across seeds ---
        print("\n--- XGBoost (tuned) ---")
        xgb_best = load_best_params(
            f"{config.MODELS_DIR}/xgb_{subset}_best_params.json")
        xgb_overrides = xgb_best.get("params", {})
        if xgb_overrides:
            print(f"Tuned overrides: {xgb_overrides}")

        xgb_runs = []
        for seed in seeds:
            params = _xgb_default_params(seed=seed)
            params.update(xgb_overrides)
            t0 = time.time()
            model = train_xgb(Xtr.values, y_tr, Xval.values, y_val,
                              params=params, verbose=0)
            y_pred = xgb_predict(model, Xte.values)
            m = both_metrics(y_te, y_pred)
            xgb_runs.append(m)
            print(f"  seed {seed:3d}: rmse={m['rmse']:.2f}  "
                  f"nasa={m['nasa_score']:.0f}  "
                  f"({time.time()-t0:.1f}s)")
            if seed == seeds[0]:
                model.save_model(
                    f"{config.MODELS_DIR}/xgb_{subset}_final_seed{seed}.json")
        results["xgboost"] = summarize_seeds(xgb_runs)

    # -----------------------------------------------------------------------
    # LSTM (raw windows)
    # -----------------------------------------------------------------------
    if model_choice in ("lstm", "all"):
        if not config._HAS_TORCH:
            print("\n=== LSTM SKIPPED (PyTorch not available) ===")
        else:
            print(f"\n=== Loading windows for {subset} ===")
            X_tr, y_tr_w, X_val, y_val_w, X_te, y_te_w = load_windows(subset)
            print(f"Window shapes: train={X_tr.shape}, val={X_val.shape}, "
                  f"test={X_te.shape}")

            print("\n--- LSTM (tuned) ---")
            lstm_best = load_best_params(
                f"{config.MODELS_DIR}/lstm_{subset}_best_params.json",
                fallback={"hidden": 64, "num_layers": 2, "dropout": 0.2, "lr": 1e-3},
            )
            # Keep only architecture/optim kwargs, discard metrics
            arch_keys = {"hidden", "num_layers", "dropout", "lr"}
            lstm_kwargs = {k: v for k, v in lstm_best.items() if k in arch_keys}
            print(f"Tuned kwargs: {lstm_kwargs}")

            lstm_runs = []
            for seed in seeds:
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass

                ckpt = f"{config.MODELS_DIR}/lstm_{subset}_final_seed{seed}.pt"
                t0 = time.time()
                model, _hist = train_lstm(
                    X_tr, y_tr_w, X_val, y_val_w,
                    epochs=100, batch=config.LSTM_BATCH,
                    patience=15, ckpt_path=ckpt,
                    **lstm_kwargs,
                )
                y_pred = lstm_predict(model, X_te)
                m = both_metrics(y_te_w, y_pred)
                lstm_runs.append(m)
                print(f"  seed {seed:3d}: rmse={m['rmse']:.2f}  "
                      f"nasa={m['nasa_score']:.0f}  "
                      f"({time.time()-t0:.1f}s)")
            results["lstm"] = summarize_seeds(lstm_runs)

    # -----------------------------------------------------------------------
    # Final table
    # -----------------------------------------------------------------------
    df = print_results_table(results, subset)

    # Save results
    out_json = f"{config.MODELS_DIR}/results_{subset}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_csv = f"{config.MODELS_DIR}/results_{subset}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--model",  default="all", choices=["xgb", "lstm", "all"])
    ap.add_argument("--seeds",  nargs="+", type=int,
                    default=[7, 13, 42, 101, 999])
    args = ap.parse_args()
    run(args.subset, args.model, args.seeds)
