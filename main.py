from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from validation.synthetic import single_vortex, double_vortex
from core.topology import compute_Qz
from core.defects import detect_defects
from tracking.worldlines import WorldlineTracker
from analysis.qzt import compute_qzt
from pipelines.run_eeg import run as run_eeg
from pipelines.run_physics import run_from_npy
from pipelines.run_cross_domain import run as run_cross_domain
from pipelines.run_physionet import run as run_physionet
from database.database import connect, start_run, finish_run, add_metric, add_artifact

def run_synthetic():
    out = Path("results/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    psi = single_vortex()
    Qz1, Qabs1 = compute_Qz(psi)
    psi2 = double_vortex()
    Qz2, Qabs2 = compute_Qz(psi2)
    df = pd.DataFrame([
        {"case":"single_vortex", "Q_mean": float(Qz1.mean()), "Qabs_mean": float(Qabs1.mean())},
        {"case":"double_vortex", "Q_mean": float(Qz2.mean()), "Qabs_mean": float(Qabs2.mean())},
    ])
    df.to_csv(out / "synthetic_summary.csv", index=False)
    print(df)

def run_qzt(checkpoint_dir: str):
    out = Path("results/qzt")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for p in sorted(Path(checkpoint_dir).rglob("psi.npy")):
        psi = np.load(p)
        meta_p = p.parent / "meta.json"
        t = float(len(checkpoints))
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text())
                t = float(meta.get("t", meta.get("step", t)))
            except Exception:
                pass
        checkpoints.append((t, psi))
    qzt = compute_qzt(checkpoints)
    qzt.to_csv(out / "qzt.csv", index=False)
    tracker = WorldlineTracker()
    for t, psi in checkpoints:
        defects = detect_defects(psi)
        tracker.update(defects, t)
    with open(out / "worldlines.json", "w", encoding="utf-8") as f:
        json.dump(tracker.get(), f, indent=2)
    print("wrote", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["synthetic", "qzt", "eeg", "physionet", "physics", "cross-domain", "db"])
    ap.add_argument("--input", default="data/checkpoints")
    ap.add_argument("--output", default="results/out.csv")
    ap.add_argument("--dataset", default="ds002094")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--compute-pci", action="store_true")
    ap.add_argument("--db", default="data/runs.sqlite")
    args = ap.parse_args()

    if args.mode == "synthetic":
        run_synthetic()
    elif args.mode == "qzt":
        run_qzt(args.input)
    elif args.mode == "eeg":
        default_out = Path("results") / args.dataset / f"metrics_{args.dataset}.csv"
        out_path = args.output if args.output != "results/out.csv" else str(default_out)
        df = run_eeg(args.input, out_path, dataset=args.dataset, compute_pci=args.compute_pci)
        print(df.head())
    elif args.mode == "physionet":
        default_out = Path("results") / "physionet_gaba" / "metrics_physionet_gaba.csv"
        out_path = args.output if args.output != "results/out.csv" else str(default_out)
        df = run_physionet(args.input, out_path)
        print(df.head())
    elif args.mode == "physics":
        df = run_from_npy(args.input, args.output)
        print(df.head())
    elif args.mode == "cross-domain":
        df = run_cross_domain(args.results_root, args.output)
        print(df.head())
    elif args.mode == "db":
        conn = connect(args.db)
        rid = start_run(conn, "manual", "demo", params={"mode": "db"})
        add_metric(conn, rid, "demo_metric", 1.0, units="arb")
        add_artifact(conn, rid, args.output, kind="csv")
        finish_run(conn, rid)
        print("db ok")

if __name__ == "__main__":
    main()
