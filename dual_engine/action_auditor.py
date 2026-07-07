"""Non-equilibrium thermodynamic auditor for the dual-engine framework.

Given a time-indexed SU(2) field (or its per-window topology results), compute:

  * F  -- a Landau-Ginzburg free-energy functional of the phase field per window:
              F = mean( kappa |grad theta|^2 + a (1 - cos theta) )
          (the (1 - cos) term is the standard XY-model periodic potential; gradients use
          the wrapped phase difference so branch cuts don't create spurious energy). This
          is a concrete, standard functional -- not a fitted or hoped-for quantity.

  * Sigma_dot -- entropy-production RATE estimated from irreversibility of a 2D collective
          observable trajectory (order-parameter magnitude R and mean phase drift). We use
          the stochastic-area (Onsager-Machlup) estimator: the mean signed area swept per
          unit time by the (x, y) trajectory. A nonzero area rate is a rigorous signature of
          broken detailed balance / time-reversal asymmetry. It is an ESTIMATOR of
          irreversibility, reported as such, not a calorimetric entropy.

  * S_Omega -- a generalized action per window, S_Omega = F + lambda * |Sigma_dot|, and its
          trend across windows. `minimizes_toward_attractor` is True iff S_Omega is
          (weakly) decreasing on average over the second half of the run. This reports a
          measured trend; it makes NO claim about what the trend means physically.

Provenance is carried through. Nothing here asserts cross-domain equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.topology import wrap_phase  # noqa: E402
from dual_engine.su2_field_mapper import SU2Field  # noqa: E402


@dataclass
class ActionAuditResult:
    F_per_window: list
    Sigma_dot: float
    S_Omega_per_window: list
    S_Omega_mean: float
    minimizes_toward_attractor: bool
    order_parameter_R: list
    provenance: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def free_energy(theta2d: np.ndarray, kappa: float = 1.0, a: float = 1.0) -> float:
    """Landau-Ginzburg / XY free-energy density of a 2D phase field (mean over pixels).

    F = mean( kappa |grad theta|^2 + a (1 - cos theta) ). Gradients use wrapped phase
    differences so a 2*pi branch cut does not register as huge energy.
    """
    theta = np.asarray(theta2d, dtype=float)
    gx = wrap_phase(np.diff(theta, axis=0))
    gy = wrap_phase(np.diff(theta, axis=1))
    grad_sq = np.mean(gx ** 2) + np.mean(gy ** 2)
    potential = np.mean(a * (1.0 - np.cos(theta)))
    return float(kappa * grad_sq + potential)


def order_parameter(theta2d: np.ndarray) -> tuple[float, float]:
    """Kuramoto-style complex order parameter: returns (R, psi) with R in [0, 1]."""
    z = np.mean(np.exp(1j * np.asarray(theta2d, dtype=float)))
    return float(np.abs(z)), float(np.angle(z))


def entropy_production_rate(traj_xy: np.ndarray, dt: float = 1.0) -> float:
    """Stochastic-area estimator of the entropy-production rate for a 2D trajectory.

    traj_xy: (T, 2) collective-observable trajectory. Returns the mean signed area swept
    per unit time = (1 / (2 dt)) * <x_t dy - y_t dx>. Nonzero => broken detailed balance
    (time-reversal asymmetry). This is an irreversibility estimator, not a calorimetric
    entropy. Returns 0.0 for T < 3.
    """
    traj = np.asarray(traj_xy, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] < 3:
        return 0.0
    x, y = traj[:, 0], traj[:, 1]
    xc, yc = x - x.mean(), y - y.mean()
    dx = np.diff(xc)
    dy = np.diff(yc)
    area = 0.5 * np.sum(xc[:-1] * dy - yc[:-1] * dx)
    T = traj.shape[0]
    return float(area / (dt * (T - 1)))


def audit_field(field_obj: SU2Field, kappa: float = 1.0, a: float = 1.0,
                lam: float = 1.0, dt: float = 1.0) -> ActionAuditResult:
    """Full thermodynamic audit of a time-indexed SU(2) field.

    Requires a 3D (T, H, W) phase field for a meaningful trajectory; a 2D field is treated
    as a single window (Sigma_dot = 0, no trajectory).
    """
    theta = field_obj.theta
    if theta.ndim == 2:
        theta = theta[None, ...]

    F_list = [free_energy(sl, kappa=kappa, a=a) for sl in theta]

    Rs, psis = [], []
    for sl in theta:
        R, psi = order_parameter(sl)
        Rs.append(R)
        psis.append(psi)

    # Collective 2D observable trajectory: (R cos psi, R sin psi) -- the mean-field vector.
    traj = np.column_stack([np.array(Rs) * np.cos(psis), np.array(Rs) * np.sin(psis)])
    sigma_dot = entropy_production_rate(traj, dt=dt)

    S_Omega = [f + lam * abs(sigma_dot) for f in F_list]

    minimizes = _is_weakly_decreasing_second_half(S_Omega)

    return ActionAuditResult(
        F_per_window=[float(x) for x in F_list],
        Sigma_dot=float(sigma_dot),
        S_Omega_per_window=[float(x) for x in S_Omega],
        S_Omega_mean=float(np.mean(S_Omega)),
        minimizes_toward_attractor=bool(minimizes),
        order_parameter_R=[float(x) for x in Rs],
        provenance=field_obj.provenance,
        meta=dict(field_obj.meta),
    )


def _is_weakly_decreasing_second_half(series: list) -> bool:
    """True iff a linear fit over the second half of the series has non-positive slope."""
    s = np.asarray(series, dtype=float)
    if len(s) < 4:
        return False
    half = s[len(s) // 2:]
    x = np.arange(len(half))
    slope = np.polyfit(x, half, 1)[0]
    return bool(slope <= 1e-9)
