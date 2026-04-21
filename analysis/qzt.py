from __future__ import annotations
from typing import Iterable, Tuple
import pandas as pd
from core.topology import compute_Qz, compute_f_dress

def compute_qzt(checkpoints: Iterable[Tuple[float, object]]) -> pd.DataFrame:
    """Compute per-slice topological summaries Q(z,t), Qabs(z,t), and f_dress(t)."""
    rows = []
    for t, psi in checkpoints:
        Qz, Qabs = compute_Qz(psi)
        f_dress = compute_f_dress(Qz, Qabs)
        for z in range(len(Qz)):
            rows.append({
                "t": float(t),
                "z": int(z),
                "Q": int(Qz[z]),
                "Qabs": float(Qabs[z]),
                "f_dress": float(f_dress),
            })
    return pd.DataFrame(rows, columns=["t", "z", "Q", "Qabs", "f_dress"])
