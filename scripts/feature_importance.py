"""
Feature-importance analysis per FD subset.

For each subset (FD001–FD004) this script:
  1. Loads the selected-feature matrix (parquet) written by extract_features.py
     and the tuned hyperparameters written by tune_xgb.py.
  2. Re-trains XGBoost on (train + val) with the tuned params.
  3. Reads the gain importance for every feature.
  4. Aggregates the importance two ways:
        a) by individual feature  (top-K table)
        b) by feature CATEGORY    (mean, std, slope, apen, baseline_*, ...)
        c) by SENSOR              (sensor_2, sensor_3, ...)
  5. Saves a CSV table and a horizontal-bar PNG plot for each subset.


Usage:
    python scripts/feature_importance.py --subset FD001
    python scripts/feature_importance.py --all
    python scripts/feature_importance.py --all --top_k 20
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import config
from models import _xgb_default_params


# ---------------------------------------------------------------------------
# Feature-name parsing
# ---------------------------------------------------------------------------
# Every tabular feature is named "<sensor>_<category>" or "<sensor>_<category>_<sub>"
# Examples: sensor_2_mean, sensor_4_slope, sensor_7_base_ks, sensor_9_ac_lag1
#
# We strip the sensor prefix to get the category, and we strip the category
# to get the sensor.  cond_* one-hots are bucketed as a single 'condition' group.
CATEGORY_REGEX = re.compile(r"^(sensor_\d+)_(.+)$")


def parse_feature_name(name: str):
    """
    Returns (sensor, category) for a feature name.
    Examples:
        sensor_2_mean         -> ('sensor_2',  'mean')
        sensor_4_slope        -> ('sensor_4',  'slope')
        sensor_7_base_ks      -> ('sensor_7',  'base_ks')
        sensor_9_ac_lag1      -> ('sensor_9',  'ac_lag1')
        cond_3                -> ('condition', 'condition')
    """
    if name.startswith("cond_"):
        return ("condition", "condition")
    m = CATEGORY_REGEX.match(name)
    if m is None:
        return ("other", name)
    return (m.group(1), m.group(2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subset(subset: str):
    """Load train + val features (with selection applied) and tuned params."""
    Ftr  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_train.parquet")
    Fval = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_val.parquet")

    # Apply selection if available (so we report importance over the same
    # feature set the final model was trained on).
    sel_path = f"{config.PROCESSED_DIR}/{subset}_selected_features.json"
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            feats = json.load(f)["features"]
        feats = [c for c in feats if c in Ftr.columns]
        Ftr  = Ftr[feats + ["RUL"]]
        Fval = Fval[feats + ["RUL"]]
        print(f"[{subset}] using {len(feats)} selected features")
    else:
        print(f"[{subset}] no selection file; using all {Ftr.shape[1] - 1} features")

    # Tuned XGBoost params
    params_path = f"{config.MODELS_DIR}/xgb_{subset}_best_params.json"
    if os.path.exists(params_path):
        with open(params_path) as f:
            tuned = json.load(f).get("params", {})
        print(f"[{subset}] loaded tuned params: {tuned}")
    else:
        tuned = {}
        print(f"[{subset}] no tuned params; using defaults")

    y_tr,  Xtr  = Ftr["RUL"].values,  Ftr.drop(columns=["RUL"])
    y_val, Xval = Fval["RUL"].values, Fval.drop(columns=["RUL"])
    return Xtr, y_tr, Xval, y_val, tuned


# ---------------------------------------------------------------------------
# Importance extraction
# ---------------------------------------------------------------------------
def train_and_get_importance(Xtr, y_tr, Xval, y_val, tuned: dict):
    """
    Train one XGBoost model with tuned params, return a DataFrame with one
    row per feature: [feature, gain, weight, cover].
    """
    params = _xgb_default_params(seed=config.SEED)
    params.update(tuned)
    feature_names = list(Xtr.columns)
    dtr  = xgb.DMatrix(Xtr.values,  label=y_tr,  feature_names=feature_names)
    dval = xgb.DMatrix(Xval.values, label=y_val, feature_names=feature_names)
    model = xgb.train(
        params, dtr,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    gain   = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    cover  = model.get_score(importance_type="cover")

    rows = []
    for f in feature_names:
        rows.append({
            "feature": f,
            "gain":    float(gain.get(f,   0.0)),
            "weight":  float(weight.get(f, 0.0)),
            "cover":   float(cover.get(f,  0.0)),
        })
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return df, model


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def aggregate_by(imp_df: pd.DataFrame, by: str = "category", normalize: bool = True):
    """
    Sum the gain across feature names that share the same group key.
    by ∈ {'category', 'sensor'}. Returns a DataFrame sorted by total gain.
    """
    parsed = imp_df["feature"].apply(parse_feature_name)
    if by == "category":
        keys = parsed.apply(lambda x: x[1])
    elif by == "sensor":
        keys = parsed.apply(lambda x: x[0])
    else:
        raise ValueError(by)
    out = (imp_df.assign(group=keys.values)
                 .groupby("group")["gain"].sum()
                 .sort_values(ascending=False)
                 .reset_index())
    if normalize and out["gain"].sum() > 0:
        out["gain_pct"] = 100.0 * out["gain"] / out["gain"].sum()
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_top_features(imp_df: pd.DataFrame, subset: str, top_k: int, out_path: str):
    """Horizontal bar plot of the top-K individual features by gain."""
    top = imp_df.head(top_k)[::-1]  # reverse so largest is at top
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.32)))
    ax.barh(top["feature"], top["gain"])
    ax.set_xlabel("Gain importance")
    ax.set_title(f"{subset} — Top {top_k} features by XGBoost gain")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_aggregated(agg_df: pd.DataFrame, subset: str, by: str, out_path: str,
                    top_k: int = 15):
    """Horizontal bar plot of the aggregated importance."""
    top = agg_df.head(top_k)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.32)))
    ax.barh(top["group"], top["gain_pct"] if "gain_pct" in top.columns else top["gain"])
    ax.set_xlabel("Gain importance (% of total)" if "gain_pct" in top.columns
                  else "Gain importance")
    ax.set_title(f"{subset} — Importance by {by}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one(subset: str, top_k: int, out_dir: str):
    print(f"\n=== Feature importance for {subset} ===")
    Xtr, y_tr, Xval, y_val, tuned = load_subset(subset)

    imp_df, _ = train_and_get_importance(Xtr, y_tr, Xval, y_val, tuned)
    by_cat = aggregate_by(imp_df, by="category")
    by_sen = aggregate_by(imp_df, by="sensor")

    os.makedirs(out_dir, exist_ok=True)
    imp_df.to_csv(   f"{out_dir}/{subset}_importance_all.csv",        index=False)
    by_cat.to_csv(   f"{out_dir}/{subset}_importance_by_category.csv", index=False)
    by_sen.to_csv(   f"{out_dir}/{subset}_importance_by_sensor.csv",   index=False)

    plot_top_features(imp_df, subset, top_k,
                      f"{out_dir}/{subset}_top{top_k}_features.png")
    plot_aggregated(  by_cat, subset, "category",
                      f"{out_dir}/{subset}_by_category.png")
    plot_aggregated(  by_sen, subset, "sensor",
                      f"{out_dir}/{subset}_by_sensor.png")

    # Console summary
    print(f"\n  Top {min(top_k, len(imp_df))} features:")
    for _, r in imp_df.head(top_k).iterrows():
        print(f"    {r['feature']:35s}  gain={r['gain']:10.2f}")
    print(f"\n  Top 10 categories (% of total gain):")
    for _, r in by_cat.head(10).iterrows():
        print(f"    {r['group']:25s}  {r['gain_pct']:5.1f}%")
    print(f"\n  Top 10 sensors (% of total gain):")
    for _, r in by_sen.head(10).iterrows():
        print(f"    {r['group']:25s}  {r['gain_pct']:5.1f}%")
    return imp_df, by_cat, by_sen


def run_all(subsets: list, top_k: int, out_dir: str):
    """Run on every subset and produce a comparison table across subsets."""
    cat_tables = {}
    for sub in subsets:
        try:
            _, by_cat, _ = run_one(sub, top_k, out_dir)
            cat_tables[sub] = by_cat.set_index("group")["gain_pct"]
        except FileNotFoundError as e:
            print(f"[{sub}] skipped: {e}")

    if not cat_tables:
        return
    combined = pd.DataFrame(cat_tables).fillna(0.0)
    combined = combined.sort_values(by=combined.columns[0], ascending=False)
    out = f"{out_dir}/category_importance_all_subsets.csv"
    combined.to_csv(out)
    print(f"\nWrote cross-subset comparison: {out}")
    print(combined.head(15).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--all", action="store_true",
                    help="Run on every FD0{01..04} subset.")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_dir", default="results/feature_importance")
    args = ap.parse_args()

    config.print_hardware()
    if args.all:
        run_all(["FD001", "FD002", "FD003", "FD004"], args.top_k, args.out_dir)
    else:
        run_one(args.subset, args.top_k, args.out_dir)
