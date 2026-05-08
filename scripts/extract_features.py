"""
Standalone feature extraction — extract once, reuse for all model experiments.

Automatically switches to condition-aware normalization for FD002 and FD004
(which have six operating conditions), and adds a one-hot condition feature
so the model can distinguish regimes.

Writes:
    data/processed/<SUBSET>_features_train.parquet
    data/processed/<SUBSET>_features_val.parquet
    data/processed/<SUBSET>_features_test.parquet
    data/processed/<SUBSET>_windows_train.npz   (raw windows for LSTM)
    data/processed/<SUBSET>_windows_val.npz
    data/processed/<SUBSET>_windows_test.npz

Usage:
    python extract_features.py --subset FD001
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import os
import numpy as np
import pandas as pd

import config
from data_loader import (
    load_subset, find_constant_sensors, add_rul_train, add_rul_test,
    minmax_scale, condition_normalize, split_engines,
    make_windows, make_last_window_per_engine,
    extract_baseline_window,
)
from features import batch_extract


def main(subset: str):
    config.print_hardware()
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    print(f"\n=== Loading {subset} ===")
    train_raw, test_raw, rul = load_subset(subset, data_dir=config.DATA_DIR)

    constant = find_constant_sensors(train_raw)
    print(f"Dropping constant sensors: {constant}")
    sensor_cols = [c for c in config.ALL_SENSORS if c not in constant]

    train_raw = add_rul_train(train_raw)
    test_raw  = add_rul_test(test_raw, rul)

    # Detect number of operating regimes from the op-setting columns.
    op_cols = ["setting_1", "setting_2", "setting_3"]
    n_regimes = len(train_raw[op_cols].round(0).drop_duplicates())
    multi_condition = n_regimes > 1
    print(f"{subset}: detected {n_regimes} operating regime(s).")

    if multi_condition:
        print(f"Using condition-aware normalization (k={n_regimes}).")
        train_raw, test_raw, _km, _scalers = condition_normalize(
            train_raw, test_raw, sensor_cols,
            n_conditions=n_regimes, seed=config.SEED)
    else:
        print("Using global minmax scaling.")
        train_raw, test_raw, _ = minmax_scale(train_raw, test_raw, sensor_cols)

    train_part, val_part = split_engines(train_raw, val_frac=0.2, seed=config.SEED)

    print("Building windows...")
    X_tr,  y_tr,  meta_tr,  _ = make_windows(train_part, sensor_cols,
                                             include_condition=multi_condition)
    X_val, y_val, meta_val, _ = make_windows(val_part,   sensor_cols,
                                             include_condition=multi_condition)
    X_te,  y_te,  ids_te, conds_te, _ = make_last_window_per_engine(
        test_raw, sensor_cols, include_condition=multi_condition)
    if conds_te is not None:
        meta_te = np.column_stack([ids_te, np.zeros_like(ids_te), conds_te])
    else:
        meta_te = np.column_stack([ids_te, np.zeros_like(ids_te)])
    print(f"Windows: train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")

    # Save raw windows for the LSTM
    np.savez_compressed(f"{config.PROCESSED_DIR}/{subset}_windows_train.npz",
                        X=X_tr, y=y_tr, meta=meta_tr)
    np.savez_compressed(f"{config.PROCESSED_DIR}/{subset}_windows_val.npz",
                        X=X_val, y=y_val, meta=meta_val)
    np.savez_compressed(f"{config.PROCESSED_DIR}/{subset}_windows_test.npz",
                        X=X_te, y=y_te, meta=meta_te)
    print("Wrote raw window .npz files.")

    baselines_train = extract_baseline_window(train_part, sensor_cols,
                                              n_cycles=config.N_BASELINE_CYCLES)
    baselines_val   = extract_baseline_window(val_part, sensor_cols,
                                              n_cycles=config.N_BASELINE_CYCLES)
    baselines_test  = extract_baseline_window(test_raw, sensor_cols,
                                              n_cycles=config.N_BASELINE_CYCLES)

    print(f"\n=== Extracting tabular features (n_jobs={config.N_JOBS}) ===")
    Ftr  = batch_extract(X_tr,  sensor_cols, meta=meta_tr,  baselines=baselines_train,
                         n_jobs=config.N_JOBS, verbose=5,
                         add_condition=multi_condition)
    Fval = batch_extract(X_val, sensor_cols, meta=meta_val, baselines=baselines_val,
                         n_jobs=config.N_JOBS,
                         add_condition=multi_condition)
    Fte  = batch_extract(X_te,  sensor_cols, meta=meta_te,  baselines=baselines_test,
                         n_jobs=config.N_JOBS,
                         add_condition=multi_condition)

    # Clean: replace inf/NaN with the train median for each column
    Ftr = Ftr.replace([np.inf, -np.inf], np.nan)
    medians = Ftr.median(numeric_only=True)
    Ftr = Ftr.fillna(medians)
    Fval = Fval.replace([np.inf, -np.inf], np.nan).fillna(medians)
    Fte  = Fte.replace([np.inf, -np.inf], np.nan).fillna(medians)

    # Align columns (train is the reference)
    Fval = Fval[Ftr.columns]
    Fte  = Fte[Ftr.columns]

    Ftr["RUL"]  = y_tr
    Fval["RUL"] = y_val
    Fte["RUL"]  = y_te

    Ftr.to_parquet(f"{config.PROCESSED_DIR}/{subset}_features_train.parquet")
    Fval.to_parquet(f"{config.PROCESSED_DIR}/{subset}_features_val.parquet")
    Fte.to_parquet(f"{config.PROCESSED_DIR}/{subset}_features_test.parquet")
    print(f"\nFeature matrix: {Ftr.shape[1] - 1} features + RUL column")
    print(f"Rows: train={len(Ftr)}, val={len(Fval)}, test={len(Fte)}")
    print(f"\nFiles written to {config.PROCESSED_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    args = ap.parse_args()
    main(args.subset)
