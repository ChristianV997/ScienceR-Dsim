from __future__ import annotations
import numpy as np
import pytest
from core.defects import detect_defects


def test_detect_defects_wrong_dim():
    with pytest.raises(ValueError):
        detect_defects(np.zeros((4, 4)))


def test_detect_defects_uniform_no_defects():
    """Uniform amplitude field has no phase winding → no defects."""
    psi = np.ones((8, 8, 4), dtype=complex)
    d = detect_defects(psi)
    assert isinstance(d, np.ndarray)
    assert len(d) == 0


def test_detect_defects_empty_return_shape():
    """Empty result must be 2-D with 4 columns."""
    psi = np.ones((8, 8, 4), dtype=complex)
    d = detect_defects(psi)
    assert d.ndim == 2
    assert d.shape == (0, 4)


def test_detect_defects_single_vortex_finds_defects():
    """A single-vortex field contains a winding defect.

    The synthetic field has unit amplitude, so the threshold must exceed 1.0
    to pass the ``amp_patch < amp_threshold`` filter.
    """
    from validation.synthetic import single_vortex
    psi = single_vortex(N=32)
    d = detect_defects(psi, amp_threshold=1.01)
    assert len(d) >= 1


def test_detect_defects_columns():
    """Returned array must have exactly four columns: x, y, z, sign."""
    from validation.synthetic import single_vortex
    psi = single_vortex(N=32)
    d = detect_defects(psi, amp_threshold=1.01)
    if len(d):
        assert d.shape[1] == 4


def test_detect_defects_signs():
    """Signs must be ±1 (not 0 or other values)."""
    from validation.synthetic import single_vortex
    psi = single_vortex(N=32)
    d = detect_defects(psi, amp_threshold=1.01)
    if len(d):
        np.testing.assert_array_equal(np.abs(d[:, 3]), 1.0)


def test_detect_defects_z_coords_in_range():
    """z-coordinate column must be within the valid z-range of the field."""
    from validation.synthetic import single_vortex
    N = 16
    psi = single_vortex(N=N)
    d = detect_defects(psi, amp_threshold=1.01)
    if len(d):
        assert np.all(d[:, 2] >= 0)
        assert np.all(d[:, 2] < N)
