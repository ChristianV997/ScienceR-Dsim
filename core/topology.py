from __future__ import annotations
import numpy as np

def wrap_phase(x: np.ndarray) -> np.ndarray:
    """Wrap phase differences into (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def plaquette_charge(theta2d: np.ndarray) -> np.ndarray:
    """Compute local winding charge on each plaquette of a 2D phase field."""
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
