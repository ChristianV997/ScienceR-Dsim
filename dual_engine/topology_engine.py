"""Unified topological calculator for the dual-engine framework.

Runs the SAME topological analysis on any SU2Field, regardless of whether it originated
from cortical EEG phase maps or a quantum field grid:

  * Q, Q_abs          -- signed and unsigned winding charge, via the already-validated
                         core.topology routines (analytic ground-truth tested to +-3 and
                         half-integer nematic charges).
  * f_dress           -- UV-dressing fraction (Q_abs - |Q|) / (|Q| + eps): excess unsigned
                         winding over coherent net charge.
  * beta1             -- first Betti number (independent loops) from REAL persistent
                         homology (ripser) on the phase field's point cloud, counting H1
                         features alive at a similarity threshold.
  * spectral_dimension-- d_s from the Laplacian eigenvalue staircase of a kNN graph over
                         the field, computed the same way for both domains.

Nothing here asserts that two fields from different domains are physically the same. It
provides identical instrumentation so their numbers are directly comparable. Provenance is
propagated onto every result.
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

from core.topology import compute_Q_slice, compute_Qabs_slice, compute_Qz, compute_f_dress  # noqa: E402
from dual_engine.su2_field_mapper import SU2Field  # noqa: E402


@dataclass
class TopologyResult:
    Q: float
    Q_abs: float
    f_dress: float
    beta1: int
    spectral_dimension: float
    n_slices: int
    provenance: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _phase_slices(field_obj: SU2Field) -> np.ndarray:
    """Return phase as (n_slices, H, W). A 2D field becomes a single slice."""
    theta = field_obj.theta
    if theta.ndim == 2:
        return theta[None, ...]
    return theta


def compute_beta1(theta2d: np.ndarray, threshold: float = 0.5, max_points: int = 400) -> int:
    """First Betti number of a phase field via REAL persistent homology (ripser).

    The phase map is embedded as a point cloud (cos theta, sin theta) per pixel; ripser
    computes H1 persistence, and beta1 counts loops alive at `threshold` (distance cutoff
    1 - threshold in the [0, 2] chord metric). Subsamples to `max_points` for tractability;
    the subsample is deterministic (evenly strided), never random, so results are
    reproducible.
    """
    import ripser

    theta = np.asarray(theta2d, dtype=float).ravel()
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    if len(pts) > max_points:
        stride = int(np.ceil(len(pts) / max_points))
        pts = pts[::stride]
    if len(pts) < 3:
        return 0
    dgms = ripser.ripser(pts, maxdim=1)["dgms"]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    if h1.size == 0:
        return 0
    d_thresh = 2.0 * (1.0 - threshold)  # chord distance for a similarity threshold
    alive = h1[(h1[:, 0] <= d_thresh) & (h1[:, 1] > d_thresh)]
    return int(alive.shape[0])


def compute_spectral_dimension(theta2d: np.ndarray, k: int = 6, max_points: int = 300) -> float:
    """Spectral dimension d_s from the Laplacian eigenvalue staircase of a kNN graph.

    Builds a symmetric k-nearest-neighbour graph over the (cos theta, sin theta) embedding,
    forms the combinatorial Laplacian, and fits d_s = 2 * slope of log N(<=lambda) vs
    log lambda over the low end of the spectrum. This is a standard operational estimator;
    it is NOT a claim about any field's true fractal dimension. Returns 0.0 when the graph
    is too small/disconnected to fit.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    theta = np.asarray(theta2d, dtype=float).ravel()
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    if len(pts) > max_points:
        stride = int(np.ceil(len(pts) / max_points))
        pts = pts[::stride]
    n = len(pts)
    if n < k + 2:
        return 0.0

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k + 1, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in idx[i][1:]:
            G.add_edge(int(i), int(j))
    if G.number_of_edges() == 0:
        return 0.0

    L = nx.laplacian_matrix(G).toarray().astype(float)
    ev = np.sort(np.linalg.eigvalsh(L))
    ev = ev[ev > 1e-9]
    if len(ev) < 4:
        return 0.0
    # cumulative eigenvalue count N(<=lambda) = rank index; fit lower half (small-lambda)
    half = max(3, len(ev) // 2)
    lam = ev[:half]
    counts = np.arange(1, half + 1)
    with np.errstate(all="ignore"):
        slope = np.polyfit(np.log(lam), np.log(counts), 1)[0]
    return float(2.0 * slope)


def analyze_field(field_obj: SU2Field, beta1_threshold: float = 0.5) -> TopologyResult:
    """Full topological analysis of an SU2Field. Provenance is carried onto the result."""
    slices = _phase_slices(field_obj)  # (n, H, W)
    n = slices.shape[0]

    # Q / Qabs per slice via validated core.topology, then aggregate.
    Qs = np.array([compute_Q_slice(sl) for sl in slices], dtype=float)
    Qabs = np.array([compute_Qabs_slice(sl) for sl in slices], dtype=float)
    f_dress = compute_f_dress(Qs, Qabs)

    # beta1 / d_s on the mean phase field (representative slice) for a single scalar summary.
    representative = slices[n // 2]
    beta1 = compute_beta1(representative, threshold=beta1_threshold)
    d_s = compute_spectral_dimension(representative)

    return TopologyResult(
        Q=float(np.mean(Qs)),
        Q_abs=float(np.mean(Qabs)),
        f_dress=float(f_dress),
        beta1=int(beta1),
        spectral_dimension=float(d_s),
        n_slices=int(n),
        provenance=field_obj.provenance,
        meta=dict(field_obj.meta),
    )


def analyze_windows(field_obj: SU2Field, beta1_threshold: float = 0.5) -> list[TopologyResult]:
    """Per-slice (per time-window) topological analysis for longitudinal tracking.

    Returns one TopologyResult per slice; each Q/Q_abs is that slice's value (n_slices=1),
    with beta1 and d_s computed on that slice.
    """
    slices = _phase_slices(field_obj)
    results = []
    for t, sl in enumerate(slices):
        Q = float(compute_Q_slice(sl))
        Qabs = float(compute_Qabs_slice(sl))
        fd = compute_f_dress(np.array([Q]), np.array([Qabs]))
        results.append(TopologyResult(
            Q=Q,
            Q_abs=Qabs,
            f_dress=float(fd),
            beta1=compute_beta1(sl, threshold=beta1_threshold),
            spectral_dimension=compute_spectral_dimension(sl),
            n_slices=1,
            provenance=field_obj.provenance,
            meta={**field_obj.meta, "window": t},
        ))
    return results
