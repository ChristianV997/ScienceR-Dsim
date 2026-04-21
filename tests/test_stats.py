from __future__ import annotations
import numpy as np
import pytest
from analysis.stats import bootstrap_ci, cohens_d, corr


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_single_element():
    lo, hi = bootstrap_ci([7.0], n=200)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_bootstrap_ci_finite_bounds():
    x = np.arange(100, dtype=float)
    lo, hi = bootstrap_ci(x, n=1000, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi


def test_bootstrap_ci_contains_true_mean():
    x = np.arange(100, dtype=float)   # true mean = 49.5
    lo, hi = bootstrap_ci(x, n=2000, seed=42)
    assert lo < 49.5 < hi


def test_bootstrap_ci_reproducible():
    x = np.arange(50, dtype=float)
    r1 = bootstrap_ci(x, seed=123)
    r2 = bootstrap_ci(x, seed=123)
    assert r1 == r2


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

def test_cohens_d_identical():
    a = np.ones(20)
    assert cohens_d(a, a) == pytest.approx(0.0, abs=1e-10)


def test_cohens_d_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    b = rng.normal(1, 1, 100)
    d = cohens_d(a, b)
    # a.mean < b.mean → negative d
    assert d < 0


def test_cohens_d_magnitude():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 500)
    b = rng.normal(1, 1, 500)
    d = cohens_d(a, b)
    # Cohen's d for Δμ=1, σ=1 should be close to -1
    assert d == pytest.approx(-1.0, abs=0.15)


def test_cohens_d_too_small():
    assert np.isnan(cohens_d([1.0], [2.0]))


# ---------------------------------------------------------------------------
# corr
# ---------------------------------------------------------------------------

def test_corr_perfect_positive():
    x = np.arange(10, dtype=float)
    result = corr(x, x)
    assert result["r"] == pytest.approx(1.0, abs=1e-10)
    assert result["p"] < 0.001


def test_corr_perfect_negative():
    x = np.arange(10, dtype=float)
    result = corr(x, -x)
    assert result["r"] == pytest.approx(-1.0, abs=1e-10)


def test_corr_too_small():
    result = corr([1.0], [1.0])
    assert np.isnan(result["r"])
    assert np.isnan(result["p"])


def test_corr_keys():
    result = corr(np.arange(5, dtype=float), np.arange(5, dtype=float))
    assert "r" in result and "p" in result
