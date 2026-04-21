from __future__ import annotations
import numpy as np

def velocity_to_phase(vx, vy):
    return np.arctan2(vy, vx)

def sample_to_psi(sample):
    vx = sample[..., 0]
    vy = sample[..., 1]
    theta = velocity_to_phase(vx, vy)
    return np.exp(1j * theta)
