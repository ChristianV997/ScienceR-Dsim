from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_qzt(df: pd.DataFrame, out: Path):
    """Render Q(z,t) heatmap figure."""
    pivot = df.pivot(index="z", columns="t", values="Q")
    plt.figure(figsize=(10, 6))
    if pivot.empty:
        plt.text(0.5, 0.5, "No Q(z,t) data", ha="center", va="center")
        plt.axis("off")
    else:
        plt.imshow(pivot, aspect="auto", origin="lower")
        plt.colorbar(label="Q")
    plt.xlabel("time")
    plt.ylabel("z")
    plt.title("Q(z,t)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_qabs(df: pd.DataFrame, out: Path):
    """Render Qabs(z,t) heatmap figure."""
    pivot = df.pivot(index="z", columns="t", values="Qabs")
    plt.figure(figsize=(10, 6))
    if pivot.empty:
        plt.text(0.5, 0.5, "No Qabs(z,t) data", ha="center", va="center")
        plt.axis("off")
    else:
        plt.imshow(pivot, aspect="auto", origin="lower")
        plt.colorbar(label="Qabs")
    plt.xlabel("time")
    plt.ylabel("z")
    plt.title("Qabs(z,t)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_worldlines(worldlines: dict, out: Path):
    """Render worldline trajectories as z-vs-time curves."""
    plt.figure(figsize=(10, 6))
    any_line = False
    for _, pts in worldlines.items():
        if len(pts) < 2:
            continue
        any_line = True
        pts = pd.DataFrame(pts, columns=["x","y","z","t"])
        plt.plot(pts["t"], pts["z"], alpha=0.6)
    if not any_line:
        plt.text(0.5, 0.5, "No worldline data", ha="center", va="center")
        plt.axis("off")
    plt.xlabel("time")
    plt.ylabel("z")
    plt.title("Defect worldlines")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def main():
    """Generate the paper figures from QZT and worldline result artifacts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    ensure(args.output_dir)

    qzt_path = args.results_root / "qzt.csv"
    world_path = args.results_root / "worldlines.json"
    if qzt_path.exists():
        df = pd.read_csv(qzt_path)
        if {"t","z","Q"}.issubset(df.columns):
            plot_qzt(df, args.output_dir / "fig_qzt.png")
        if {"t","z","Qabs"}.issubset(df.columns):
            plot_qabs(df, args.output_dir / "fig_qabs.png")
    else:
        plot_qzt(pd.DataFrame(columns=["t", "z", "Q"]), args.output_dir / "fig_qzt.png")
        plot_qabs(pd.DataFrame(columns=["t", "z", "Qabs"]), args.output_dir / "fig_qabs.png")
    if world_path.exists():
        world = json.loads(world_path.read_text())
        plot_worldlines(world, args.output_dir / "fig_worldlines.png")
    else:
        plot_worldlines({}, args.output_dir / "fig_worldlines.png")

if __name__ == "__main__":
    main()
