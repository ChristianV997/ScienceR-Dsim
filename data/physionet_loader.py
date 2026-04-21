from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def load_physionet_gaba_folder(root: str | Path):
    root = Path(root)
    rows = []
    for p in root.rglob("*.csv"):
        name = p.name
        if not any(k in name for k in ["alphaWave", "slowWave", "val_spec", "t_spec", "f_spec", "duration_start_end"]):
            continue
        try:
            arr = pd.read_csv(p, header=None).values.astype(float).flatten()
        except Exception:
            continue
        if len(arr) < 4:
            continue
        rows.append({"file": str(p), "n": len(arr), "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))})
    return pd.DataFrame(rows)
