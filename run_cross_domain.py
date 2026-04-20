from __future__ import annotations
import numpy as np

def single_vortex(N=64):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    psi = np.exp(1j * theta)
    return np.repeat(psi[:, :, None], N, axis=2)

def double_vortex(N=64):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    theta1 = np.arctan2(Y - 0.25, X - 0.25)
    theta2 = np.arctan2(Y + 0.25, X + 0.25)
    psi = np.exp(1j * (theta1 + theta2))
    return np.repeat(psi[:, :, None], N, axis=2)
