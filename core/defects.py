from __future__ import annotations
import numpy as np
from .topology import plaquette_charge

def detect_defects(psi3d: np.ndarray, amp_threshold: float = 0.2):
    """Locate candidate defects by winding charge and low local amplitude."""
    psi3d = np.asarray(psi3d)
    if psi3d.ndim != 3:
        raise ValueError("psi3d must be 3D")
    defects = []
    for z in range(psi3d.shape[2]):
        sl = psi3d[:, :, z]
        theta = np.angle(sl)
        amp = np.abs(sl)
        q = plaquette_charge(theta)
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                if abs(q[i, j]) > 0.5 and amp[i:i+2, j:j+2].mean() < amp_threshold:
                    defects.append([i, j, z, int(np.sign(q[i, j]))])
    return np.asarray(defects, dtype=float)
