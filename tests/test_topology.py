from __future__ import annotations
import numpy as np
import pytest
from core.topology import (
    wrap_phase,
    plaquette_charge,
    compute_Q_slice,
    compute_Qabs_slice,
    compute_Qz,
    compute_f_dress,
)


# ---------------------------------------------------------------------------
# wrap_phase
# ---------------------------------------------------------------------------

def test_wrap_phase_identity():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    np.testing.assert_allclose(wrap_phase(x), x)


def test_wrap_phase_large_positive():
    # 2π wraps to ~0
    assert abs(wrap_phase(2 * np.pi)) < 1e-10


def test_wrap_phase_minus_pi():
    # -π and +π represent the same angle; the formula maps -π → -π which is
    # still a valid boundary value with |result| == π.
    result = wrap_phase(-np.pi)
    assert abs(result) == pytest.approx(np.pi)


def test_wrap_phase_array():
    x = np.array([-np.pi - 0.1, np.pi + 0.1])
    result = wrap_phase(x)
    assert np.all(result > -np.pi)
    assert np.all(result <= np.pi)


# ---------------------------------------------------------------------------
# plaquette_charge
# ---------------------------------------------------------------------------

def test_plaquette_charge_uniform():
    theta = np.zeros((8, 8))
    q = plaquette_charge(theta)
    assert q.shape == (7, 7)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)


def test_plaquette_charge_shape():
    theta = np.random.default_rng(0).normal(size=(10, 12))
    q = plaquette_charge(theta)
    assert q.shape == (9, 11)


def test_plaquette_charge_wrong_dim():
    with pytest.raises(ValueError):
        plaquette_charge(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# compute_Q_slice / compute_Qabs_slice
# ---------------------------------------------------------------------------

def test_compute_Q_slice_uniform():
    assert compute_Q_slice(np.zeros((8, 8))) == 0


def test_compute_Qabs_slice_uniform():
    assert compute_Qabs_slice(np.zeros((8, 8))) == 0.0


def test_compute_Qabs_slice_nonnegative():
    rng = np.random.default_rng(1)
    theta = rng.uniform(-np.pi, np.pi, (16, 16))
    assert compute_Qabs_slice(theta) >= 0.0


# ---------------------------------------------------------------------------
# compute_Qz
# ---------------------------------------------------------------------------

def test_compute_Qz_wrong_dim():
    with pytest.raises(ValueError):
        compute_Qz(np.zeros((4, 4)))


def test_compute_Qz_shape():
    psi = np.ones((4, 4, 5), dtype=complex)
    Qz, Qabs = compute_Qz(psi)
    assert Qz.shape == (5,)
    assert Qabs.shape == (5,)


def test_compute_Qz_axis0():
    psi = np.ones((5, 4, 4), dtype=complex)
    Qz, Qabs = compute_Qz(psi, axis=0)
    assert Qz.shape == (5,)
    assert Qabs.shape == (5,)


def test_compute_Qz_uniform_zero():
    psi = np.ones((8, 8, 4), dtype=complex)
    Qz, Qabs = compute_Qz(psi)
    np.testing.assert_array_equal(Qz, 0)
    np.testing.assert_allclose(Qabs, 0.0, atol=1e-12)


def test_compute_Qz_single_vortex():
    from validation.synthetic import single_vortex
    psi = single_vortex(N=32)
    Qz, Qabs = compute_Qz(psi)
    assert Qz.shape == (32,)
    assert float(np.mean(Qz)) == pytest.approx(1.0, abs=0.1)


def test_compute_Qz_double_vortex():
    from validation.synthetic import double_vortex
    psi = double_vortex(N=32)
    Qz, Qabs = compute_Qz(psi)
    assert float(np.mean(Qz)) == pytest.approx(2.0, abs=0.25)


# ---------------------------------------------------------------------------
# compute_f_dress
# ---------------------------------------------------------------------------

def test_compute_f_dress_coherent():
    """Coherent case: |mean(Qz)| == mean(Qabs) → f_dress ≈ 0."""
    Qz = np.array([1, 1, 1])
    Qabs = np.array([1.0, 1.0, 1.0])
    assert compute_f_dress(Qz, Qabs) == pytest.approx(0.0, abs=1e-6)


def test_compute_f_dress_incoherent():
    """If Qabs dominates over net charge, f_dress > 0."""
    Qz = np.array([0, 0, 0])
    Qabs = np.array([2.0, 2.0, 2.0])
    assert compute_f_dress(Qz, Qabs) > 0


def test_compute_f_dress_nonnegative():
    rng = np.random.default_rng(0)
    Qz = rng.integers(-3, 4, size=20)
    Qabs = np.abs(Qz.astype(float)) + rng.uniform(0, 0.5, size=20)
    assert compute_f_dress(Qz, Qabs) >= 0
