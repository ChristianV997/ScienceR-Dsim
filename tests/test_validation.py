from __future__ import annotations
import numpy as np
import pytest
from validation.synthetic import single_vortex, double_vortex, validate_vortex_charges


def test_single_vortex_shape():
    psi = single_vortex(N=32)
    assert psi.shape == (32, 32, 32)


def test_double_vortex_shape():
    psi = double_vortex(N=32)
    assert psi.shape == (32, 32, 32)


def test_single_vortex_dtype():
    psi = single_vortex(N=8)
    assert np.issubdtype(psi.dtype, np.complexfloating)


def test_single_vortex_unit_amplitude():
    psi = single_vortex(N=8)
    np.testing.assert_allclose(np.abs(psi), 1.0, atol=1e-10)


def test_double_vortex_unit_amplitude():
    psi = double_vortex(N=8)
    np.testing.assert_allclose(np.abs(psi), 1.0, atol=1e-10)


def test_validate_vortex_charges_both_pass():
    result = validate_vortex_charges()
    assert result["single_vortex_pass"], (
        f"single-vortex failed: Q_mean={result['single_vortex_Q_mean']}"
    )
    assert result["double_vortex_pass"], (
        f"double-vortex failed: Q_mean={result['double_vortex_Q_mean']}"
    )


def test_validate_vortex_charges_values():
    result = validate_vortex_charges()
    assert result["single_vortex_Q_mean"] == pytest.approx(1.0, abs=0.25)
    assert result["double_vortex_Q_mean"] == pytest.approx(2.0, abs=0.25)


def test_validate_vortex_charges_keys():
    result = validate_vortex_charges()
    for key in (
        "single_vortex_Q_mean",
        "double_vortex_Q_mean",
        "single_vortex_pass",
        "double_vortex_pass",
    ):
        assert key in result
