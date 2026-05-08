"""
Unit tests for data_loader.py — RUL target construction and windowing.
"""
import numpy as np
import pandas as pd
import pytest

from data_loader import (
    add_rul_train, add_rul_test, find_constant_sensors, make_windows,
)
from config import ALL_SENSORS, R_EARLY


def _toy_train_df(n_engines=3, n_cycles=50):
    """Build a toy training frame with the right column layout."""
    rows = []
    rng = np.random.default_rng(0)
    for uid in range(1, n_engines + 1):
        for c in range(1, n_cycles + 1):
            row = {"unit_id": uid, "cycle": c,
                   "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
            for s in ALL_SENSORS:
                row[s] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RUL target construction
# ---------------------------------------------------------------------------
def test_add_rul_train_correct_at_endpoints():
    df = _toy_train_df(n_engines=2, n_cycles=10)
    out = add_rul_train(df.copy())
    # First cycle of engine 1 should have RUL = max_cycle - 1 = 9.
    first = out[(out["unit_id"] == 1) & (out["cycle"] == 1)].iloc[0]
    assert first["RUL"] == 9
    # Last cycle should have RUL = 0.
    last = out[(out["unit_id"] == 1) & (out["cycle"] == 10)].iloc[0]
    assert last["RUL"] == 0


def test_add_rul_train_clipped_at_R_early():
    df = _toy_train_df(n_engines=1, n_cycles=200)
    out = add_rul_train(df.copy())
    # All RUL_clipped values must be <= R_EARLY.
    assert out["RUL_clipped"].max() == R_EARLY
    # Late cycles must not be clipped (they are below R_EARLY).
    last_rows = out.tail(5)
    assert (last_rows["RUL_clipped"] == last_rows["RUL"]).all()


def test_add_rul_test_uses_external_rul():
    test_df = _toy_train_df(n_engines=2, n_cycles=10)
    rul_df = pd.DataFrame({"RUL": [50, 30]})
    out = add_rul_test(test_df.copy(), rul_df)
    last_e1 = out[(out["unit_id"] == 1) & (out["cycle"] == 10)].iloc[0]
    # End-of-test RUL = the value from RUL_FD00x.txt.
    assert last_e1["RUL"] == 50
    last_e2 = out[(out["unit_id"] == 2) & (out["cycle"] == 10)].iloc[0]
    assert last_e2["RUL"] == 30


# ---------------------------------------------------------------------------
# Constant-sensor detection
# ---------------------------------------------------------------------------
def test_find_constant_sensors():
    df = _toy_train_df(n_engines=1, n_cycles=50)
    df["sensor_1"] = 5.0           # truly constant
    df["sensor_5"] = 1e-7          # below threshold
    consts = find_constant_sensors(df, threshold=1e-4)
    assert "sensor_1" in consts and "sensor_5" in consts


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_make_windows_shapes():
    df = _toy_train_df(n_engines=2, n_cycles=50)
    df = add_rul_train(df)
    sensors = ALL_SENSORS[:5]
    X, y, meta, _ = make_windows(df, sensors, window=30, stride=1)
    # Each engine: 50 - 30 + 1 = 21 windows; two engines: 42.
    assert X.shape == (42, 30, 5)
    assert y.shape == (42,)
    assert meta.shape[0] == 42


def test_make_windows_targets_match_last_cycle_rul():
    df = _toy_train_df(n_engines=1, n_cycles=40)
    df = add_rul_train(df)
    sensors = ALL_SENSORS[:3]
    X, y, meta, _ = make_windows(df, sensors, window=30, stride=1)
    # First window covers cycles 1..30 -> last cycle is 30 -> RUL = 40 - 30 = 10
    # but R_EARLY clipping doesn't apply because RUL is small.
    assert y[0] == 10
