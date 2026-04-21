from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr

def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    # Vectorised: draw all (n × len(x)) indices at once to avoid a Python loop
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / (sp + 1e-12))

def corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"r": np.nan, "p": np.nan}
    r, p = pearsonr(a, b)
    return {"r": float(r), "p": float(p)}
