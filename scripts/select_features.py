"""
Feature selection: compare three methods, pick the best on validation set.

Run AFTER extract_features.py has written the parquet files.

Usage:
    python select_features.py --subset FD001
    python select_features.py --subset FD001 --method xgb_importance --k 150
    python select_features.py --subset FD001 --compare
"""
import _path  # noqa: F401  # adds src/ to sys.path
import argparse
import json
import os
import numpy as np
import pandas as pd

import config
from evaluate import rmse, nasa_score


def load_features(subset: str):
    """Load the three parquet files produced by extract_features.py."""
    Ftr  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_train.parquet")
    Fval = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_val.parquet")
    Fte  = pd.read_parquet(f"{config.PROCESSED_DIR}/{subset}_features_test.parquet")
    y_tr  = Ftr["RUL"].values;  Xtr  = Ftr.drop(columns=["RUL"])
    y_val = Fval["RUL"].values; Xval = Fval.drop(columns=["RUL"])
    y_te  = Fte["RUL"].values;  Xte  = Fte.drop(columns=["RUL"])
    return Xtr, y_tr, Xval, y_val, Xte, y_te


# ---------------------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------------------
def select_pearson(Xtr, y_tr, k: int):
    """Top-k features by absolute Pearson correlation with the target."""
    scores = np.abs(Xtr.apply(lambda col: np.corrcoef(col, y_tr)[0, 1]).values)
    top_idx = np.argsort(scores)[-k:]
    return list(Xtr.columns[top_idx]), scores


def select_mutual_info(Xtr, y_tr, k: int):
    """Top-k features by mutual information with the target."""
    from sklearn.feature_selection import mutual_info_regression
    scores = mutual_info_regression(Xtr, y_tr, random_state=config.SEED)
    top_idx = np.argsort(scores)[-k:]
    return list(Xtr.columns[top_idx]), scores


def select_xgb_importance(Xtr, y_tr, Xval, y_val, k: int):
    """Train a quick XGBoost, keep top-k by gain importance."""
    import xgboost as xgb
    from models import _xgb_default_params
    params = _xgb_default_params(seed=config.SEED)
    dtr  = xgb.DMatrix(Xtr.values,  label=y_tr,  feature_names=list(Xtr.columns))
    dval = xgb.DMatrix(Xval.values, label=y_val, feature_names=list(Xval.columns))
    model = xgb.train(params, dtr, num_boost_round=500,
                      evals=[(dval, "val")],
                      early_stopping_rounds=50, verbose_eval=False)
    imp = model.get_score(importance_type="gain")
    # Missing features get zero importance
    scores = np.array([imp.get(c, 0.0) for c in Xtr.columns])
    top_idx = np.argsort(scores)[-k:]
    return list(Xtr.columns[top_idx]), scores


def evaluate_subset(Xtr, y_tr, Xval, y_val, Xte, y_te, feat_list):
    """Train XGBoost on the selected feature list, return val and test metrics."""
    import xgboost as xgb
    from models import _xgb_default_params
    feat_list = _bundle_conditions(feat_list, Xtr.columns)

    params = _xgb_default_params(seed=config.SEED)
    dtr  = xgb.DMatrix(Xtr[feat_list].values,  label=y_tr)
    dval = xgb.DMatrix(Xval[feat_list].values, label=y_val)
    dte  = xgb.DMatrix(Xte[feat_list].values,  label=y_te)
    model = xgb.train(params, dtr, num_boost_round=1000,
                      evals=[(dval, "val")],
                      early_stopping_rounds=50, verbose_eval=False)
    y_pred_val = model.predict(dval)
    y_pred_te  = model.predict(dte)
    return {
        "val_rmse": rmse(y_val, y_pred_val),
        "val_nasa": nasa_score(y_val, y_pred_val),
        "test_rmse": rmse(y_te, y_pred_te),
        "test_nasa": nasa_score(y_te, y_pred_te),
    }


def compare_methods(subset: str, k_values=(217, 218, 219, 220, 221, 222, 223, 224)):
    """Run all three methods at several k values; report a comparison table."""
    Xtr, y_tr, Xval, y_val, Xte, y_te = load_features(subset)
    print(f"\nFull feature count: {Xtr.shape[1]}")

    # Baseline: all features
    all_feats = list(Xtr.columns)
    base = evaluate_subset(Xtr, y_tr, Xval, y_val, Xte, y_te, all_feats)
    print(f"\nAll features ({len(all_feats)}): val_rmse={base['val_rmse']:.2f}  test_rmse={base['test_rmse']:.2f}")

    results = [{"method": "all", "k": len(all_feats), **base}]

    for method_name, selector in [
        ##("pearson", lambda k: select_pearson(Xtr, y_tr, k)[0]),
        ##("mutual_info", lambda k: select_mutual_info(Xtr, y_tr, k)[0]),
        ("xgb_importance",
         lambda k: select_xgb_importance(Xtr, y_tr, Xval, y_val, k)[0]),
    ]:
        for k in k_values:
            if k > Xtr.shape[1]:
                continue
            feat_list = selector(k)
            m = evaluate_subset(Xtr, y_tr, Xval, y_val, Xte, y_te, feat_list)
            print(f"{method_name:16s} k={k:3d}: val_rmse={m['val_rmse']:.2f}  test_rmse={m['test_rmse']:.2f}")
            results.append({"method": method_name, "k": k, **m})

    df = pd.DataFrame(results).sort_values("val_rmse")
    print("\n=== Ranked by validation RMSE ===")
    print(df.to_string(index=False))
    out = f"{config.PROCESSED_DIR}/{subset}_selection_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Report the recommended configuration
    best = df.iloc[0]
    print(f"\nBEST: {best['method']} with k={best['k']} (val_rmse={best['val_rmse']:.2f})")
    return df


def save_selection(subset: str, method: str, k: int):
    """Save the chosen feature list to a JSON file so train_xgb/lstm can use it."""
    Xtr, y_tr, Xval, y_val, Xte, y_te = load_features(subset)
    if method == "pearson":
        feats, scores = select_pearson(Xtr, y_tr, k)
    elif method == "mutual_info":
        feats, scores = select_mutual_info(Xtr, y_tr, k)
    elif method == "xgb_importance":
        feats, scores = select_xgb_importance(Xtr, y_tr, Xval, y_val, k)
    else:
        raise ValueError(f"unknown method {method}")

    feats = _bundle_conditions(feats, Xtr.columns)

    out = f"{config.PROCESSED_DIR}/{subset}_selected_features.json"
    with open(out, "w") as f:
        json.dump({"method": method, "k": k, "features": feats}, f, indent=2)
    print(f"Saved {len(feats)} selected features to {out}")

def _bundle_conditions(feat_list, all_cols):
    """If any cond_* column is kept, keep them all together."""
    cond_cols = [c for c in all_cols if c.startswith("cond_")]
    if cond_cols and any(c in feat_list for c in cond_cols):
        feat_list = list(dict.fromkeys(list(feat_list) + cond_cols))
    return feat_list

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--method", default="xgb_importance",
                    choices=["pearson", "mutual_info", "xgb_importance"])
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--compare", action="store_true",
                    help="Run all methods at multiple k values.")
    args = ap.parse_args()

    if args.compare:
        compare_methods(args.subset)
    else:
        save_selection(args.subset, args.method, args.k)
