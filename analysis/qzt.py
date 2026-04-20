from __future__ import annotations
from typing import Iterable, Tuple
import pandas as pd
from core.topology import compute_Qz

def compute_qzt(checkpoints: Iterable[Tuple[float, object]]) -> pd.DataFrame:
    """Compute per-slice topological summaries Q(z,t) and Qabs(z,t)."""
    rows = []
    for t, psi in checkpoints:
        Qz, Qabs = compute_Qz(psi)
        for z in range(len(Qz)):
            rows.append({"t": float(t), "z": int(z), "Q": int(Qz[z]), "Qabs": float(Qabs[z])})
    return pd.DataFrame(rows, columns=["t", "z", "Q", "Qabs"])
