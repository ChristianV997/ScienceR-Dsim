from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from data.the_well_loader import sample_to_psi
from core.topology import compute_Qz

def run_from_npy(npy_path: str | Path, output_csv: str | Path):
    npy_path = Path(npy_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sample = np.load(npy_path)
    psi = sample_to_psi(sample)
    psi = np.repeat(psi[:, :, None], 8, axis=2)

    Qz, Qabs = compute_Qz(psi)
    df = pd.DataFrame({"z": np.arange(len(Qz)), "Q": Qz, "Qabs": Qabs})
    df.to_csv(output_csv, index=False)
    return df
