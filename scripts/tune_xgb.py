"""
Sequential hyperparameter tuning for XGBoost.

Tunes greedily, one hyperparameter at a time:
  1. max_depth
  2. learning_rate
  3. min_child_weight
  4. subsample + colsample_bytree (together)
  5. reg_lambda

Uses the feature selection JSON from select_features.py. GPU is auto-used
via _xgb_default_params.

Usage:
    python tune_xgb.py --subset FD001
    python tune_xgb.py --subset FD002 --use_all_features
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from evaluate import rmse, nasa_score
from models import _xgb_default_params


def load_features(subset: str, use_selection: bool = True):
    """Load parquet features and apply selection if available."""
    Ftr  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_train.parquet")
    Fval = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_val.parquet")
    Fte  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_test.parquet")

    y_tr, y_val, y_te = Ftr["RUL"].values, Fval["RUL"].values, Fte["RUL"].values
    Xtr  = Ftr.drop(columns=["RUL"])
    Xval = Fval.drop(columns=["RUL"])
    Xte  = Fte.drop(columns=["RUL"])

    if use_selection:
        sel_path = f"{config.PROCESSED_DIR}/{subset}_selected_features.json"
        if os.path.exists(sel_path):
            with open(sel_path) as f:
                feats = json.load(f)["features"]
            feats = [c for c in feats if c in Xtr.columns]
            Xtr, Xval, Xte = Xtr[feats], Xval[feats], Xte[feats]
            print(f"Using {len(feats)} selected features from {sel_path}")
        else:
            print(f"WARNING: {sel_path} not found. Using all features.")

    print(f"Shapes: train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}")
    return Xtr, y_tr, Xval, y_val, Xte, y_te


def run_trial(Xtr, y_tr, Xval, y_val, overrides,
              num_boost_round=5000, early_stop=100):
    """Train one XGBoost model with the given overrides. Returns val metrics."""
    params = _xgb_default_params(seed=config.SEED)
    params.update(overrides)
    dtr  = xgb.DMatrix(Xtr.values,  label=y_tr)
    dval = xgb.DMatrix(Xval.values, label=y_val)
    t0 = time.time()
    model = xgb.train(
        params, dtr,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    elapsed = time.time() - t0
    y_pred = model.predict(dval)
    return {
        "val_rmse": rmse(y_val, y_pred),
        "val_nasa": nasa_score(y_val, y_pred),
        "best_iter": int(model.best_iteration),
        "time_s": round(elapsed, 1),
    }


def tune_sequential(subset: str, use_selection: bool):
    """Stage-by-stage greedy tuning."""
    Xtr, y_tr, Xval, y_val, _, _ = load_features(subset, use_selection=use_selection)

    results = []
    best = {}  # winning hyperparameters accumulate here

    stages = [
        # Stage 1: Tree Architecture
        ("max_depth", [4, 6, 8, 10]),
        ("min_child_weight", [1, 3, 5, 10]),

        # Stage 2: Pruning
        ("gamma", [0, 0.1, 1.0, 5.0]),

        # Stage 3: Randomization
        ("subsample", [0.7, 0.85, 1.0]),
        ("colsample_bytree", [0.7, 0.85, 1.0]),

        # Stage 4: Regularization
        ("reg_lambda", [0.5, 1.0, 5.0, 10.0]),  # L2
        ("reg_alpha", [0, 0.1, 1.0, 5.0]),  # L1

        # Stage 5: Learning Rate
        ("learning_rate", [0.01, 0.03, 0.05, 0.1]),
    ]

    for stage_name, values in stages:
        print(f"\n=== Tuning {stage_name} ===")
        stage_trials = []
        for v in values:
            if stage_name == "sampling":
                overrides = {**best, "subsample": v[0], "colsample_bytree": v[1]}
                desc = f"subsample={v[0]}, colsample={v[1]}"
            else:
                overrides = {**best, stage_name: v}
                desc = f"{stage_name}={v}"

            r = run_trial(Xtr, y_tr, Xval, y_val, overrides)
            stage_trials.append((v, r))
            results.append({"stage": stage_name, "value": str(v), **r})
            print(f"  {desc}:  val_rmse={r['val_rmse']:.3f}  "
                  f"val_nasa={r['val_nasa']:.0f}  best_iter={r['best_iter']}  "
                  f"time={r['time_s']}s")

        # Lock in the best value for this stage
        v_best, r_best = min(stage_trials, key=lambda t: t[1]["val_rmse"])
        if stage_name == "sampling":
            best["subsample"], best["colsample_bytree"] = v_best
            print(f"  -> best: subsample={v_best[0]}, colsample={v_best[1]}  "
                  f"(val_rmse={r_best['val_rmse']:.3f})")
        else:
            best[stage_name] = v_best
            print(f"  -> best: {stage_name}={v_best}  "
                  f"(val_rmse={r_best['val_rmse']:.3f})")

    print(f"\n=== Final config ===")
    final = run_trial(Xtr, y_tr, Xval, y_val, best)
    print(f"best params: {best}")
    print(f"final val_rmse: {final['val_rmse']:.3f}  "
          f"val_nasa: {final['val_nasa']:.0f}  best_iter: {final['best_iter']}")

    out_json = f"{config.MODELS_DIR}/xgb_{subset}_best_params.json"
    with open(out_json, "w") as f:
        json.dump({
            "subset": subset,
            "params": best,
            "val_rmse": final["val_rmse"],
            "val_nasa": final["val_nasa"],
            "best_iter": final["best_iter"],
        }, f, indent=2)
    print(f"Wrote {out_json}")

    out_csv = f"{config.MODELS_DIR}/xgb_{subset}_tuning_log.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--use_all_features", action="store_true",
                    help="Skip selected_features.json and use all features.")
    args = ap.parse_args()

    config.print_hardware()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    tune_sequential(args.subset, use_selection=not args.use_all_features)