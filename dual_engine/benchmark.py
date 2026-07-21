"""Side-by-side dual-engine collision benchmark (reality-first, no equivalence asserted).

Runs the SAME instrumentation (topology_engine + action_auditor) on two clearly-labelled
proxies and reports their numbers next to each other:

  Test 1 -- Phase-Boundary Collapse:
      neuro proxy   : a cessation-like collapse of a cortical phase field (order parameter
                      decays across windows), provenance="synthetic_proxy".
      quantum proxy : a topological quench of a phase grid (winding annihilates across
                      windows), provenance="quantum_field".
      We report per-window Q, beta1, S_Omega and Sigma_dot for both. We DO NOT assert they
      lie on the same non-Hermitian Exceptional Point trajectory -- the output explicitly
      records `equivalence_asserted: false` and lists what real-data test would be required.

  Test 2 -- Multi-Scale Fractality:
      We report the measured spectral dimension d_s of each proxy's connectivity graph. We
      DO NOT claim convergence to any target value (e.g. 2.10); that is an empirical claim
      requiring real ds003969 / gauge-lattice data. `target_convergence_claimed: false`.

Outputs a provenance-stamped JSON and (if matplotlib present) a side-by-side figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dual_engine.su2_field_mapper import phase_to_su2  # noqa: E402
from dual_engine.topology_engine import analyze_windows  # noqa: E402
from dual_engine.action_auditor import audit_field  # noqa: E402


def _neuro_cessation_proxy(n=48, n_windows=10, seed=0) -> np.ndarray:
    """Cortical phase field whose collective coherence collapses across windows.

    Early windows: a coherent +1 vortex (organized boundary). Late windows: coherence
    decays into spatial noise (boundary dissolves). SYNTHETIC -- labelled synthetic_proxy.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n) - n / 2 + 0.37
    y, x = np.meshgrid(idx, idx, indexing="ij")
    vortex = np.arctan2(y, x)
    frames = []
    for w in range(n_windows):
        coherence = max(0.0, 1.0 - w / (n_windows - 1))  # 1 -> 0
        noise = (1.0 - coherence) * np.pi * rng.standard_normal((n, n))
        frames.append((coherence * vortex + noise + np.pi) % (2 * np.pi) - np.pi)
    return np.stack(frames, axis=0)


def _quantum_quench_proxy(n=48, n_windows=10, seed=1) -> np.ndarray:
    """Phase grid whose winding annihilates across windows (a topological quench).

    Early windows: a +1/-1 vortex pair (net charge 0, high unsigned winding). The pair
    approaches and annihilates, reducing unsigned winding to ~0. SYNTHETIC quantum_field.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n) - n / 2 + 0.37
    y, x = np.meshgrid(idx, idx, indexing="ij")
    frames = []
    for w in range(n_windows):
        sep = (1.0 - w / (n_windows - 1)) * 0.3 * n  # pair separation shrinks to 0
        cx1, cx2 = -sep, sep
        th = np.arctan2(y, x - cx1) - np.arctan2(y, x - cx2)
        th = th + 0.02 * rng.standard_normal((n, n))
        frames.append((th + np.pi) % (2 * np.pi) - np.pi)
    return np.stack(frames, axis=0)


def run_benchmark(out_dir: str = "outputs/dual_engine", n_windows: int = 10) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    neuro = phase_to_su2(_neuro_cessation_proxy(n_windows=n_windows),
                         provenance="synthetic_proxy",
                         meta={"proxy": "neuro_cessation"})
    quantum = phase_to_su2(_quantum_quench_proxy(n_windows=n_windows),
                           provenance="quantum_field",
                           meta={"proxy": "quantum_quench"})

    neuro_topo = [r.to_dict() for r in analyze_windows(neuro)]
    quantum_topo = [r.to_dict() for r in analyze_windows(quantum)]
    neuro_audit = audit_field(neuro).to_dict()
    quantum_audit = audit_field(quantum).to_dict()

    payload = {
        "benchmark": "dual_engine_collision_v1",
        "equivalence_asserted": False,
        "target_convergence_claimed": False,
        "note": (
            "Numbers are computed with identical instrumentation on two SYNTHETIC proxies. "
            "No physical equivalence between the neuro and quantum systems is asserted, and "
            "no convergence of spectral dimension to any target value is claimed. Any such "
            "claim requires real ds003969/ds000245 EEG and real gauge/BEC field data run "
            "through NeuroBIDSAdapter and QuantumFieldAdapter respectively."
        ),
        "test1_phase_boundary_collapse": {
            "neuro": {"provenance": "synthetic_proxy",
                      "Q_per_window": [r["Q"] for r in neuro_topo],
                      "Qabs_per_window": [r["Q_abs"] for r in neuro_topo],
                      "beta1_per_window": [r["beta1"] for r in neuro_topo],
                      "S_Omega_per_window": neuro_audit["S_Omega_per_window"],
                      "Sigma_dot": neuro_audit["Sigma_dot"],
                      "R_per_window": neuro_audit["order_parameter_R"]},
            "quantum": {"provenance": "quantum_field",
                        "Q_per_window": [r["Q"] for r in quantum_topo],
                        "Qabs_per_window": [r["Q_abs"] for r in quantum_topo],
                        "beta1_per_window": [r["beta1"] for r in quantum_topo],
                        "S_Omega_per_window": quantum_audit["S_Omega_per_window"],
                        "Sigma_dot": quantum_audit["Sigma_dot"],
                        "R_per_window": quantum_audit["order_parameter_R"]},
        },
        "test2_multiscale_fractality": {
            "neuro_spectral_dimension_per_window": [r["spectral_dimension"] for r in neuro_topo],
            "quantum_spectral_dimension_per_window": [r["spectral_dimension"] for r in quantum_topo],
            "target_value_2_10_claimed": False,
        },
    }

    (out / "collision_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _maybe_plot(out, payload)
    return payload


def _maybe_plot(out: Path, payload: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    t1 = payload["test1_phase_boundary_collapse"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for lab, key in (("neuro (synthetic_proxy)", "neuro"), ("quantum (quantum_field)", "quantum")):
        d = t1[key]
        axes[0].plot(d["Qabs_per_window"], marker="o", label=lab)
        axes[1].plot(d["S_Omega_per_window"], marker="s", label=lab)
        axes[2].plot(d["R_per_window"], marker="^", label=lab)
    axes[0].set_title("Unsigned winding Q_abs vs window")
    axes[1].set_title(r"Generalized action $S_\Omega$ vs window")
    axes[2].set_title("Order parameter R vs window")
    for ax in axes:
        ax.set_xlabel("window"); ax.legend(fontsize=8)
    fig.suptitle("Dual-engine collision benchmark (SYNTHETIC proxies; no equivalence asserted)")
    fig.tight_layout()
    fig.savefig(out / "collision_benchmark.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-engine collision benchmark")
    ap.add_argument("--out", default="outputs/dual_engine")
    ap.add_argument("--n-windows", type=int, default=10)
    args = ap.parse_args(argv)
    payload = run_benchmark(args.out, n_windows=args.n_windows)
    t1 = payload["test1_phase_boundary_collapse"]
    print(f"equivalence_asserted={payload['equivalence_asserted']} "
          f"target_convergence_claimed={payload['target_convergence_claimed']}")
    print(f"neuro   Qabs[0->-1]={t1['neuro']['Qabs_per_window'][0]:.2f}->"
          f"{t1['neuro']['Qabs_per_window'][-1]:.2f} Sigma_dot={t1['neuro']['Sigma_dot']:.4f}")
    print(f"quantum Qabs[0->-1]={t1['quantum']['Qabs_per_window'][0]:.2f}->"
          f"{t1['quantum']['Qabs_per_window'][-1]:.2f} Sigma_dot={t1['quantum']['Sigma_dot']:.4f}")
    print(f"-> outputs in {args.out}/collision_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
