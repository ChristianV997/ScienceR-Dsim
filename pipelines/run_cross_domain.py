from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run(results_root: str | Path, output_csv: str | Path):
    results_root = Path(results_root)
    output_csv = Path(output_csv)
    rows = []

    for ds in ["ds002094", "ds005620", "ds001787", "ds003969", "ds003816"]:
        f = results_root / ds / f"metrics_{ds}.csv"
        if f.exists():
            df = pd.read_csv(f)
            if len(df):
                rows.append({
                    "dataset": ds,
                    "domain": "brain",
                    "Q": float(df["Q"].mean()) if "Q" in df.columns else None,
                    "Qabs": float(df["Qabs"].mean()) if "Qabs" in df.columns else None,
                    "phase_grad": float(df["phase_grad"].mean()) if "phase_grad" in df.columns else None,
                })

    f = results_root / "physionet_gaba" / "metrics_physionet_gaba.csv"
    if f.exists():
        df = pd.read_csv(f)
        rows.append({
            "dataset": "physionet_gaba",
            "domain": "brain_like_signal",
            "Q": float(df["Q_proxy"].mean()) if "Q_proxy" in df.columns else None,
            "Qabs": float(df["Qabs_proxy"].mean()) if "Qabs_proxy" in df.columns else None,
            "phase_grad": None,
        })

    f = results_root / "the_well" / "metrics_the_well.csv"
    if f.exists():
        df = pd.read_csv(f)
        rows.append({
            "dataset": "the_well",
            "domain": "physics",
            "Q": float(df["Q"].mean()) if "Q" in df.columns else None,
            "Qabs": float(df["Qabs"].mean()) if "Qabs" in df.columns else None,
            "phase_grad": None,
        })

    df = pd.DataFrame(rows).dropna()
    if len(df):
        X = df[["Q", "Qabs"]].values
        X = StandardScaler().fit_transform(X)
        df["cluster"] = KMeans(n_clusters=min(3, len(df)), random_state=42, n_init=10).fit_predict(X)
    df.to_csv(output_csv, index=False)
    return df
