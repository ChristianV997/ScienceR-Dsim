"""Tests for the dual-engine topological framework (dual_engine/*).

Covers SU(2) mapping, unified topology engine, both data adapters, and the thermodynamic
auditor. All fixtures are synthetic and labelled; no real dataset or network required.
ripser is required for beta1 (skips cleanly if absent).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dual_engine.su2_field_mapper import (
    phase_to_su2, su2_to_phase, is_in_su2, stack_windows_to_field, SU2Field,
)
from dual_engine.action_auditor import (
    free_energy, order_parameter, entropy_production_rate, audit_field,
)

_HAVE_RIPSER = importlib.util.find_spec("ripser") is not None
_HAVE_H5PY = importlib.util.find_spec("h5py") is not None


def _vortex(n=64, charge=1.0, ox=0.37, oy=0.81):
    idx_y = (np.arange(n) - n / 2 + oy)
    idx_x = (np.arange(n) - n / 2 + ox)
    y, x = np.meshgrid(idx_y, idx_x, indexing="ij")
    return charge * np.arctan2(y, x)


# --------------------------------------------------------------------------- SU(2)

def test_su2_lift_is_unitary_det_one():
    f = phase_to_su2(_vortex(), provenance="synthetic_proxy")
    assert is_in_su2(f.U)


def test_su2_phase_roundtrip_exact():
    theta = _vortex()
    f = phase_to_su2(theta, provenance="synthetic_proxy")
    rec = su2_to_phase(f)
    # compare mod 2pi
    err = np.max(np.abs(np.angle(np.exp(1j * (rec - f.theta)))))
    assert err < 1e-9


def test_su2_rejects_bad_provenance():
    with pytest.raises(ValueError):
        phase_to_su2(_vortex(), provenance="made_up")


def test_su2_general_axis_still_in_su2():
    f = phase_to_su2(_vortex(), axis=np.array([1.0, 1.0, 1.0]), provenance="quantum_field")
    assert is_in_su2(f.U)


def test_stack_windows_shape():
    ws = [_vortex(), _vortex(charge=-1.0)]
    f = stack_windows_to_field(ws, provenance="synthetic_proxy")
    assert f.theta.shape == (2, 64, 64)
    assert f.U.shape == (2, 64, 64, 2, 2)


# --------------------------------------------------------------------- topology engine

@pytest.mark.skipif(not _HAVE_RIPSER, reason="requires ripser")
def test_topology_recovers_vortex_charge_and_loop():
    from dual_engine.topology_engine import analyze_field
    f = phase_to_su2(_vortex(charge=1.0), provenance="synthetic_proxy")
    res = analyze_field(f)
    # documented sign convention: +1 field -> Q = -1
    assert res.Q == pytest.approx(-1.0)
    assert res.Q_abs == pytest.approx(1.0)
    assert res.f_dress == pytest.approx(0.0, abs=1e-6)
    assert res.beta1 >= 1  # a real H1 loop exists in the phase circle embedding
    assert res.provenance == "synthetic_proxy"


@pytest.mark.skipif(not _HAVE_RIPSER, reason="requires ripser")
def test_topology_per_window_carries_provenance():
    from dual_engine.topology_engine import analyze_windows
    ws = [_vortex(charge=1.0), _vortex(charge=2.0)]
    f = stack_windows_to_field(ws, provenance="quantum_field")
    results = analyze_windows(f)
    assert len(results) == 2
    assert all(r.provenance == "quantum_field" for r in results)
    assert results[1].Q == pytest.approx(-2.0)  # +2 field -> Q = -2


# ------------------------------------------------------------------------ auditor

def test_free_energy_nonnegative_and_finite():
    f = free_energy(_vortex())
    assert f >= 0 and np.isfinite(f)


def test_order_parameter_bounds():
    R, psi = order_parameter(_vortex())
    assert 0.0 <= R <= 1.0


def test_entropy_production_zero_for_reversible_line():
    # a back-and-forth line sweeps zero net area -> reversible
    t = np.linspace(0, 1, 50)
    traj = np.column_stack([np.concatenate([t, t[::-1]]), np.zeros(100)])
    assert abs(entropy_production_rate(traj)) < 1e-9


def test_entropy_production_nonzero_for_circulating_loop():
    # A closed circulating loop sweeps nonzero area -> irreversible signature.
    # The estimator returns mean signed area PER STEP; a unit circle has area pi over
    # (T-1) steps, so the rate is ~pi/(T-1). It must be clearly nonzero and clearly
    # larger than the reversible back-and-forth line (which is ~0).
    t = np.linspace(0, 2 * np.pi, 200)
    circulating = np.column_stack([np.cos(t), np.sin(t)])
    line_t = np.linspace(0, 1, 50)
    reversible = np.column_stack([np.concatenate([line_t, line_t[::-1]]), np.zeros(100)])
    circ_rate = abs(entropy_production_rate(circulating))
    rev_rate = abs(entropy_production_rate(reversible))
    assert circ_rate > 1e-3           # clearly nonzero irreversibility
    assert circ_rate > 100 * (rev_rate + 1e-12)  # and far above the reversible baseline
    # sanity: matches the analytic area-per-step of a unit circle, pi/(T-1)
    assert circ_rate == pytest.approx(np.pi / 199, rel=0.05)


def test_audit_field_reports_trend_and_provenance():
    stack = np.stack([_vortex() + 0.2 * k for k in range(8)], axis=0)
    f = phase_to_su2(stack, provenance="synthetic_proxy")
    res = audit_field(f)
    assert len(res.F_per_window) == 8
    assert isinstance(res.minimizes_toward_attractor, bool)
    assert res.provenance == "synthetic_proxy"


# ------------------------------------------------------------------- data adapters

@pytest.mark.skipif(not _HAVE_H5PY, reason="requires h5py")
def test_quantum_field_adapter_complex(tmp_path):
    import h5py
    from dual_engine.data_adapters import QuantumFieldAdapter
    # a synthetic complex spinor grid with a planted phase winding
    theta = np.stack([_vortex(charge=1.0), _vortex(charge=-1.0)], axis=0)
    fieldc = np.exp(1j * theta)
    p = tmp_path / "qfield.h5"
    with h5py.File(p, "w") as fh:
        fh.create_dataset("psi", data=fieldc)
    adapter = QuantumFieldAdapter(str(p))
    su2 = adapter.load_su2_field(field_key="psi")
    assert su2.provenance == "quantum_field"
    assert su2.theta.shape == (2, 64, 64)


@pytest.mark.skipif(not _HAVE_H5PY, reason="requires h5py")
def test_quantum_field_adapter_missing_file_raises():
    from dual_engine.data_adapters import QuantumFieldAdapter
    adapter = QuantumFieldAdapter("/nonexistent/path.h5")
    with pytest.raises(FileNotFoundError):
        adapter.load_phase_windows()


def test_neuro_adapter_requires_real_bids_import():
    # NeuroBIDSAdapter reuses data.bids_ingest; just confirm it constructs and the
    # dependency is importable (no network / no real data needed for this check).
    from dual_engine.data_adapters import NeuroBIDSAdapter
    a = NeuroBIDSAdapter("/some/bids/root")
    assert a.provenance == "real_bids"
