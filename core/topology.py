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
    """Compute Q and Qabs for each slice along a selected axis of a 3D field."""
    psi3d = np.asarray(psi3d)
    if psi3d.ndim != 3:
        raise ValueError("psi3d must be 3D")
    sl = np.moveaxis(psi3d, axis, 0)
    Qz, Qabs = [], []
    for s in sl:
        theta = np.angle(s)
        Qz.append(compute_Q_slice(theta))
        Qabs.append(compute_Qabs_slice(theta))
    return np.asarray(Qz, dtype=int), np.asarray(Qabs, dtype=float)

def compute_f_dress(Qz: np.ndarray, Qabs: np.ndarray, eps: float = 1e-9) -> float:
    """Compute a simple excess-absolute-charge ratio summary statistic."""
    return float((np.mean(Qabs) - abs(np.mean(Qz))) / (abs(np.mean(Qz)) + eps))
