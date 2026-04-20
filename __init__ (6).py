from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def pcist_surrogate(epoch_2d):
    x = np.mean(epoch_2d, axis=0)
    z = (x - np.mean(x)) / (np.std(x) + 1e-12)
    b = (z > 0).astype(np.int8)
    transitions = np.sum(np.abs(np.diff(b)))
    active = np.mean(b)
    return float(transitions + active)

def q_pcist_correlation(df: pd.DataFrame):
    if "Qabs" not in df.columns or "PCIst" not in df.columns:
        return {"r": np.nan, "p": np.nan}
    sub = df[["Qabs", "PCIst"]].dropna()
    if len(sub) < 3:
        return {"r": np.nan, "p": np.nan, "n": int(len(sub))}
    r, p = pearsonr(sub["Qabs"], sub["PCIst"])
    return {"r": float(r), "p": float(p), "n": int(len(sub))}
