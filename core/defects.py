from __future__ import annotations
import numpy as np
from .topology import wrap_phase

def detect_defects(psi3d: np.ndarray, amp_threshold: float = 0.2):
    """Locate candidate defects by winding charge and low local amplitude.

    Vectorised implementation: plaquette charges and amplitude patch-means are
    computed for all (x, y, z) positions simultaneously, replacing the O(Nx·Ny)
    Python inner loop with a single NumPy masking operation per z-slice.
    The function now always returns a 2-D array of shape ``(N, 4)`` where the
    four columns are ``[x, y, z, sign]``; N is 0 when no defects are found.
    """
    psi3d = np.asarray(psi3d)
    if psi3d.ndim != 3:
        raise ValueError("psi3d must be 3D")

    Nx, Ny, Nz = psi3d.shape
    theta = np.angle(psi3d)   # (Nx, Ny, Nz)
    amp   = np.abs(psi3d)     # (Nx, Ny, Nz)

    # Plaquette charges for every (x, y, z) simultaneously — shape (Nx-1, Ny-1, Nz)
    a = wrap_phase(theta[1:, :-1, :] - theta[:-1, :-1, :])
    b = wrap_phase(theta[1:, 1:,  :] - theta[1:, :-1, :])
    c = wrap_phase(theta[:-1, 1:, :] - theta[1:, 1:, :])
    d = wrap_phase(theta[:-1, :-1, :] - theta[:-1, 1:, :])
    q = (a + b + c + d) / (2 * np.pi)            # (Nx-1, Ny-1, Nz)

    # 2×2 amplitude patch mean at each plaquette corner — shape (Nx-1, Ny-1, Nz)
    amp_patch = (
        amp[:-1, :-1, :] + amp[1:, :-1, :]
        + amp[:-1, 1:, :] + amp[1:, 1:, :]
    ) / 4.0

    mask = (np.abs(q) > 0.5) & (amp_patch < amp_threshold)
    ii, jj, zz = np.where(mask)
    if len(ii) == 0:
        return np.empty((0, 4), dtype=float)
    signs = np.sign(q[ii, jj, zz]).astype(int)
    return np.column_stack([ii, jj, zz, signs]).astype(float)
