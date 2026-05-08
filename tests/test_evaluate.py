"""
Unit tests for evaluate.py.

These are the most safety-critical functions in the project: a bug in the
NASA scoring function would silently make every result wrong. We pin them
against hand-computed values.
"""
import math

import numpy as np
import pytest

from evaluate import rmse, nasa_score, both_metrics


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------
class TestRMSE:
    def test_zero_when_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self):
        # errors = [1, -1, 1, -1], mean of squares = 1, sqrt = 1.
        assert rmse([0, 0, 0, 0], [1, -1, 1, -1]) == pytest.approx(1.0)

    def test_handles_lists_and_arrays(self):
        a = rmse([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        b = rmse(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
        assert a == pytest.approx(b)

    def test_symmetric(self):
        # Same magnitude of error in either direction yields the same RMSE.
        assert rmse([0, 0], [+5, -5]) == rmse([0, 0], [-5, +5])


# ---------------------------------------------------------------------------
# NASA asymmetric score
# ---------------------------------------------------------------------------
class TestNASAScore:
    def test_zero_when_perfect(self):
        assert nasa_score([1, 2, 3], [1, 2, 3]) == 0.0

    def test_late_more_costly_than_early(self):
        # Predicting 10 cycles late should cost more than 10 cycles early.
        early = nasa_score([100], [90])  # d = -10  -> exp(10/13) - 1
        late  = nasa_score([100], [110]) # d = +10  -> exp(10/10) - 1
        assert late > early
        # Numeric pin: early ≈ 1.1604, late ≈ 1.7183
        assert early == pytest.approx(math.exp(10 / 13.0) - 1, rel=1e-9)
        assert late  == pytest.approx(math.exp(10 / 10.0) - 1, rel=1e-9)

    def test_sums_over_samples(self):
        # Two perfectly-late predictions should equal 2 * one of them.
        s1 = nasa_score([50],     [60])
        s2 = nasa_score([50, 50], [60, 60])
        assert s2 == pytest.approx(2 * s1)

    def test_lower_is_better(self):
        # Closer prediction must score strictly lower.
        good = nasa_score([100], [101])
        bad  = nasa_score([100], [120])
        assert good < bad


def test_both_metrics_returns_both():
    out = both_metrics([1, 2, 3], [1, 2, 3])
    assert "rmse" in out and "nasa_score" in out
    assert out["rmse"] == 0.0 and out["nasa_score"] == 0.0
