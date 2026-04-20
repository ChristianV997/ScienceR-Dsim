from __future__ import annotations
import numpy as np
from core.topology import compute_Qz

def single_vortex(N=64):
    """Create a synthetic single-vortex complex field."""
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    theta = -np.arctan2(Y, X)
    psi = np.exp(1j * theta)
    return np.repeat(psi[:, :, None], N, axis=2)

def double_vortex(N=64):
    """Create a synthetic double-vortex complex field."""
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    theta1 = -np.arctan2(Y - 0.25, X - 0.25)
    theta2 = -np.arctan2(Y + 0.25, X + 0.25)
    psi = np.exp(1j * (theta1 + theta2))
    return np.repeat(psi[:, :, None], N, axis=2)

def validate_vortex_charges(tol: float = 0.25) -> dict:
    """Validate expected synthetic charges for single and double vortex fields."""
    q1, _ = compute_Qz(single_vortex())
    q2, _ = compute_Qz(double_vortex())
    q1_mean = float(np.mean(q1))
    q2_mean = float(np.mean(q2))
    return {
        "single_vortex_Q_mean": q1_mean,
        "double_vortex_Q_mean": q2_mean,
        "single_vortex_pass": bool(abs(q1_mean - 1.0) <= tol),
        "double_vortex_pass": bool(abs(q2_mean - 2.0) <= tol),
    }
