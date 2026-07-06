from __future__ import annotations
import numpy as np

def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap phase differences into (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def plaquette_charge(theta2d: np.ndarray) -> np.ndarray:
    """Compute local winding charge on each plaquette of a 2D phase field.

    Sign convention: the (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1) loop used here traces
    clockwise when axis 0 is plotted as the vertical (y) axis increasing upward and
    axis 1 as horizontal (x). This gives the NEGATIVE of the standard counter-clockwise
    mathematical winding number: a true charge +k field yields Q=-k here. Validated
    against analytic ground-truth defects (integer charges -3..+3, half-integer nematic
    charges via the 2-theta convention) across resolutions 16-2048 and multiple
    non-degenerate core offsets -- see tests/test_topology_analytic_ground_truth.py.
    """
    if theta2d.ndim != 2:
        raise ValueError("theta2d must be 2D")
    a = wrap_phase(theta2d[1:, :-1] - theta2d[:-1, :-1])
    b = wrap_phase(theta2d[1:, 1:] - theta2d[1:, :-1])
    c = wrap_phase(theta2d[:-1, 1:] - theta2d[1:, 1:])
    d = wrap_phase(theta2d[:-1, :-1] - theta2d[:-1, 1:])
    return (a + b + c + d) / (2 * np.pi)

def compute_Q_slice(theta2d: np.ndarray) -> int:
    """Compute integer-like signed topological charge Q for one 2D slice."""
    q = plaquette_charge(theta2d)
    return int(np.rint(q.sum()))


def compute_nematic_defect_charge(director_theta2d: np.ndarray) -> float:
    """Topological charge of a nematic director field (head-tail symmetric, defined mod pi).

    Nematic defects are physically half-integer (+-1/2, +-1, +-3/2, ...) because the
    director n and -n are physically identical. Applying plaquette_charge directly to
    the raw director angle is WRONG for half-integer defects (it silently returns 0 for
    a true +-1/2 defect). The correct approach doubles the angle so the field becomes
    single-valued mod 2*pi, computes the integer winding of that, then halves the result.
    Sign-corrected to the standard counter-clockwise mathematical convention (opposite of
    plaquette_charge's raw sign -- see plaquette_charge docstring).
    """
    psi = wrap_phase(2.0 * director_theta2d)
    q_psi = compute_Q_slice(psi)
    return -q_psi / 2.0

def compute_Qabs_slice(theta2d: np.ndarray) -> float:
    """Compute absolute plaquette charge sum Qabs for one 2D slice."""
    q = plaquette_charge(theta2d)
    return float(np.sum(np.abs(q)))

def compute_Qz(psi3d: np.ndarray, axis: int = 2):
    """Compute Q and Qabs for each slice along a selected axis of a 3D field.

    Vectorised implementation: operates on all slices simultaneously instead
    of looping in Python, giving a substantial speed-up for large fields.
    """
    psi3d = np.asarray(psi3d)
    if psi3d.ndim != 3:
        raise ValueError("psi3d must be 3D")
    # Move the target axis to position 0 so shape is (nslices, nx, ny)
    sl = np.moveaxis(psi3d, axis, 0)
    theta = np.angle(sl)                         # (nslices, nx, ny)
    a = wrap_phase(theta[:, 1:, :-1] - theta[:, :-1, :-1])
    b = wrap_phase(theta[:, 1:, 1:]  - theta[:, 1:, :-1])
    c = wrap_phase(theta[:, :-1, 1:] - theta[:, 1:, 1:])
    d = wrap_phase(theta[:, :-1, :-1] - theta[:, :-1, 1:])
    q = (a + b + c + d) / (2 * np.pi)           # (nslices, nx-1, ny-1)
    Qz   = np.rint(q.sum(axis=(1, 2))).astype(int)
    Qabs = np.abs(q).sum(axis=(1, 2))
    return Qz, Qabs

def compute_f_dress(Qz: np.ndarray, Qabs: np.ndarray, eps: float = 1e-9) -> float:
    """Compute excess absolute winding relative to net winding.

    This ratio is near zero when local positive/negative winding mostly cancels
    to a coherent net topological charge, and increases when unsigned local
    winding dominates over the signed net charge.
    """
    return float((np.mean(Qabs) - abs(np.mean(Qz))) / (abs(np.mean(Qz)) + eps))
