from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from analysis.qzt import compute_qzt
from analysis.events import detect_events


def _vortex_checkpoints(n=2, N=16):
    from validation.synthetic import single_vortex
    psi = single_vortex(N=N)
    return [(float(i), psi) for i in range(n)]


# ---------------------------------------------------------------------------
# compute_qzt
# ---------------------------------------------------------------------------

def test_compute_qzt_empty():
    df = compute_qzt([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_compute_qzt_columns():
    df = compute_qzt(_vortex_checkpoints(n=2))
    assert {"t", "z", "Q", "Qabs", "f_dress"}.issubset(df.columns)


def test_compute_qzt_row_count():
    N = 16
    df = compute_qzt(_vortex_checkpoints(n=3, N=N))
    # Each checkpoint contributes N z-rows
    assert len(df) == 3 * N


def test_compute_qzt_f_dress_nonnegative():
    df = compute_qzt(_vortex_checkpoints(n=2))
    assert (df["f_dress"] >= 0).all()


def test_compute_qzt_Q_dtype():
    df = compute_qzt(_vortex_checkpoints(n=1))
    # Q should be integer-valued
    assert pd.api.types.is_integer_dtype(df["Q"]) or df["Q"].apply(float.is_integer).all()


# ---------------------------------------------------------------------------
# detect_events
# ---------------------------------------------------------------------------

def test_detect_events_empty_df():
    ev = detect_events(pd.DataFrame())
    assert isinstance(ev, pd.DataFrame)
    assert len(ev) == 0


def test_detect_events_missing_cols():
    ev = detect_events(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert len(ev) == 0


def test_detect_events_no_change():
    data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "z": [0, 0, 0], "Q": [1, 1, 1]})
    ev = detect_events(data)
    assert len(ev) == 0


def test_detect_events_single_change():
    data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "z": [0, 0, 0], "Q": [1, 1, 2]})
    ev = detect_events(data)
    assert len(ev) == 1
    assert int(ev.iloc[0]["delta"]) == 1


def test_detect_events_multiple_z():
    data = pd.DataFrame({
        "t": [0.0, 1.0, 0.0, 1.0],
        "z": [0, 0, 1, 1],
        "Q": [0, 1, 2, 2],
    })
    ev = detect_events(data)
    # Only z=0 changes
    assert len(ev) == 1
    assert int(ev.iloc[0]["z"]) == 0


def test_detect_events_columns():
    data = pd.DataFrame({"t": [0.0, 1.0], "z": [0, 0], "Q": [0, 1]})
    ev = detect_events(data)
    assert {"z", "t", "delta"}.issubset(ev.columns)
