"""
Unit tests for features.py — checks that each feature returns sensible values
on a few known inputs.
"""
import numpy as np
import pytest

from features import (
    basic_stats, shape_stats, quantile_stats, trend_stats,
    autocorr_stats, approximate_entropy, hurst_exponent,
    cusum_max, baseline_comparison, tail_variance_ratio,
    extract_sensor_features,
)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
def test_basic_stats_constant_signal():
    out = basic_stats(np.full(30, 7.0))
    assert out["mean"] == 7.0
    assert out["std"]  == 0.0
    assert out["range"] == 0.0


def test_basic_stats_linear_signal():
    out = basic_stats(np.arange(10, dtype=float))
    assert out["mean"] == pytest.approx(4.5)
    assert out["min"]  == 0.0
    assert out["max"]  == 9.0


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def test_trend_perfect_increasing_line():
    # A perfectly linear ramp must give slope = 1 and r2 = 1.
    out = trend_stats(np.arange(30, dtype=float))
    assert out["slope"] == pytest.approx(1.0)
    assert out["r2"]    == pytest.approx(1.0)


def test_trend_flat_signal_zero_slope():
    out = trend_stats(np.full(30, 5.0))
    assert out["slope"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------
def test_autocorr_lag1_close_to_one_for_smooth_signal():
    # A slow ramp is highly autocorrelated. The exact value for a 30-point ramp
    # is just under 0.9 (≈ 0.8999...) because the formula divides by sum over
    # the full window rather than the (W-k)-truncated one.
    out = autocorr_stats(np.arange(30, dtype=float))
    assert out["ac_lag1"] > 0.85


def test_autocorr_lag1_near_zero_for_white_noise():
    rng = np.random.default_rng(0)
    out = autocorr_stats(rng.standard_normal(500))
    assert abs(out["ac_lag1"]) < 0.2


# ---------------------------------------------------------------------------
# Entropy & Hurst
# ---------------------------------------------------------------------------
def test_apen_more_for_noise_than_constant():
    rng = np.random.default_rng(0)
    apen_const = approximate_entropy(np.full(60, 1.0))
    apen_rand  = approximate_entropy(rng.standard_normal(60))
    assert apen_rand > apen_const


def test_hurst_above_half_for_trending_signal():
    h = hurst_exponent(np.arange(80, dtype=float))
    # A monotone ramp should be persistent (H > 0.5).
    assert h > 0.5


# ---------------------------------------------------------------------------
# CUSUM and tail variance
# ---------------------------------------------------------------------------
def test_cusum_max_zero_for_constant_signal():
    assert cusum_max(np.full(30, 4.0)) == pytest.approx(0.0)


def test_tail_variance_ratio_high_when_late_volatility_high():
    rng = np.random.default_rng(0)
    early = np.zeros(20)
    late  = rng.standard_normal(20) * 5
    sig   = np.concatenate([early, late])
    assert tail_variance_ratio(sig) > 1.0


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------
def test_baseline_comparison_zero_when_identical():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(30)
    out = baseline_comparison(w, w)
    assert out["base_ks"]         == pytest.approx(0.0, abs=1e-9)
    assert out["base_mean_shift"] == pytest.approx(0.0, abs=1e-9)
    # The implementation adds an epsilon to the denominator to prevent
    # divide-by-zero, so the ratio is 1 to ~7 decimals, not bit-exact.
    assert out["base_var_ratio"]  == pytest.approx(1.0, rel=1e-6)


def test_baseline_comparison_detects_mean_shift():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(30)
    now  = base + 5.0  # large mean shift
    out  = baseline_comparison(now, base)
    assert out["base_mean_shift"] > 4.0


# ---------------------------------------------------------------------------
# End-to-end per-sensor extractor
# ---------------------------------------------------------------------------
def test_extract_sensor_features_naming():
    feats = extract_sensor_features(np.arange(30, dtype=float), prefix="sensor_2")
    # All keys must be prefixed with the sensor name.
    assert all(k.startswith("sensor_2_") for k in feats)
    # And we must get a non-trivial number of features.
    assert len(feats) >= 25
