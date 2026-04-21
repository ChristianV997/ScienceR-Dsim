from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from validation.pci_validation import pcist_surrogate, q_pcist_correlation


# ---------------------------------------------------------------------------
# pcist_surrogate
# ---------------------------------------------------------------------------

def test_pcist_surrogate_returns_float():
    epoch = np.random.default_rng(0).standard_normal((10, 100))
    v = pcist_surrogate(epoch)
    assert isinstance(v, float)


def test_pcist_surrogate_finite():
    epoch = np.random.default_rng(1).standard_normal((5, 50))
    assert np.isfinite(pcist_surrogate(epoch))


def test_pcist_surrogate_constant_epoch():
    """Constant epoch has no transitions; result should still be finite."""
    epoch = np.ones((4, 30))
    assert np.isfinite(pcist_surrogate(epoch))


def test_pcist_surrogate_nonnegative():
    """Result is a count of transitions plus an active-fraction; must be ≥ 0."""
    epoch = np.random.default_rng(2).standard_normal((8, 64))
    assert pcist_surrogate(epoch) >= 0


# ---------------------------------------------------------------------------
# q_pcist_correlation
# ---------------------------------------------------------------------------

def test_q_pcist_correlation_missing_cols():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = q_pcist_correlation(df)
    assert np.isnan(result["r"])
    assert np.isnan(result["p"])


def test_q_pcist_correlation_too_few_rows():
    df = pd.DataFrame({"Qabs": [1.0, 2.0], "PCIst": [0.5, 1.0]})
    result = q_pcist_correlation(df)
    assert result["n"] == 2


def test_q_pcist_correlation_perfect():
    rng = np.random.default_rng(42)
    x = rng.random(30)
    df = pd.DataFrame({"Qabs": x, "PCIst": x + rng.normal(0, 0.005, 30)})
    result = q_pcist_correlation(df)
    assert result["r"] > 0.99
    assert result["n"] == 30


def test_q_pcist_correlation_keys():
    df = pd.DataFrame({"Qabs": range(5), "PCIst": range(5)})
    result = q_pcist_correlation(df)
    assert {"r", "p", "n"}.issubset(result.keys())
