"""Evaluation metrics: RMSE and the NASA asymmetric scoring function."""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt( (1/N) sum_i (y_pred_i - y_true_i)^2 )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """
    NASA asymmetric scoring function (Saxena et al., 2008).

    Let d_i = y_pred_i - y_true_i. Then:
        s_i = exp(-d_i / 13) - 1   if d_i < 0   (predicted early)
        s_i = exp( d_i / 10) - 1   if d_i >= 0  (predicted late)
    Total score S = sum_i s_i. Lower is better.

    Late predictions are penalized harder than early ones — you want
    maintenance scheduled before the real failure cycle, not after.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(s))


def both_metrics(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred),
            "nasa_score": nasa_score(y_true, y_pred)}


def summarize_runs(all_metrics: list) -> dict:
    """Aggregate metrics from multiple seeds: returns mean and std."""
    import pandas as pd
    df = pd.DataFrame(all_metrics)
    return {
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std":  float(df["rmse"].std(ddof=1)),
        "nasa_mean": float(df["nasa_score"].mean()),
        "nasa_std":  float(df["nasa_score"].std(ddof=1)),
    }
