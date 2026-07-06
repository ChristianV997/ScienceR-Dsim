"""Validate core/topology.py's defect-charge detection against The Well's active_matter
dataset (Polymathic AI) -- real, published biological active-nematic simulation data,
independent of any EEG or consciousness framing.

Real dataset schema (pulled from github.com/PolymathicAI/the_well, not fabricated):
  - 256x256 grid, periodic BC, 81 timesteps per trajectory, 16 files, real physical
    parameters (alpha, zeta, L) swept across simulations.
  - Field "D" is the nematic orientation tensor: D_xx, D_xy, D_yx, D_yy. This is the
    real physical quantity whose defects this script measures -- NOT an EEG phase field,
    NOT a consciousness signature. See docs/datasets/active_matter.md in that repo and
    the associated paper (arXiv:2308.06675) for the physical model.

The nematic director angle is recovered from the D-tensor via the standard formula
theta = 0.5 * atan2(2*D_xy, D_xx - D_yy), then core.topology.compute_nematic_defect_charge
(2-theta convention) gives the physical defect charge per frame.

Requires network access to HuggingFace (blocked in this build sandbox -- see
NETWORK_BOUNDARY note below). Run on a machine with that access for real validation;
--synthetic-check runs a schema-faithful offline smoke test instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.topology import compute_nematic_defect_charge  # noqa: E402

NETWORK_BOUNDARY = (
    "This sandbox's egress blocks huggingface.co (403), so real active_matter arrays "
    "cannot be pulled from here. The schema above is real (pulled from the source repo, "
    "not fabricated); the numeric validation below needs to run on a networked machine."
)


def director_from_D_tensor(D_xx: np.ndarray, D_xy: np.ndarray, D_yy: np.ndarray) -> np.ndarray:
    """Standard nematic director angle from the Q/D order-parameter tensor components."""
    return 0.5 * np.arctan2(2.0 * D_xy, D_xx - D_yy)


def defect_charges_over_time(D_xx: np.ndarray, D_xy: np.ndarray, D_yy: np.ndarray) -> list[float]:
    """D_xx/D_xy/D_yy shape (T, H, W). Returns net topological charge per frame.

    Note: for a field with multiple defects of opposite sign, the *net* charge over
    the whole frame will often be ~0 even when many individual defects are present
    (they're created/annihilate in +/- pairs). This is expected physics, not a bug --
    a manuscript-grade defect census would localize individual defects, not just sum
    the whole-frame net charge. That localization is a follow-up, not done here.
    """
    charges = []
    for t in range(D_xx.shape[0]):
        theta = director_from_D_tensor(D_xx[t], D_xy[t], D_yy[t])
        charges.append(compute_nematic_defect_charge(theta))
    return charges


def _synthetic_schema_faithful_check(n_frames: int = 6, grid: int = 256, seed: int = 0) -> dict:
    """Offline smoke test: builds a 256x256 field with a KNOWN, planted +1/-1 nematic
    defect PAIR (net charge exactly 0 by construction, the physically expected case for
    a closed/periodic domain) and confirms the pipeline recovers that net charge. This
    is schema-faithful (same grid size, same D-tensor field names/shape as real
    active_matter) but the signal is synthetic -- labelled as such, not real Well data.
    """
    rng = np.random.default_rng(seed)
    frames = []
    idx = np.arange(grid) - grid / 2 + 0.37
    y, x = np.meshgrid(idx, idx, indexing="ij")
    # Defect centers defined in the SAME centered/shifted coordinate frame as idx
    # (roughly -grid/2..+grid/2), not raw pixel-index units -- mixing frames here was
    # a real bug caught during review: it silently misplaced the second defect's core.
    cx1, cy1 = -0.2 * grid, -0.2 * grid
    cx2, cy2 = 0.2 * grid, 0.1 * grid
    for t in range(n_frames):
        theta1 = 0.5 * np.arctan2(y - cy1, x - cx1)
        theta2 = -0.5 * np.arctan2(y - cy2, x - cx2)
        director = theta1 + theta2 + 0.02 * rng.standard_normal((grid, grid))  # small real-ish noise
        D_xx = np.cos(2 * director)
        D_xy = np.sin(2 * director)
        D_yy = -np.cos(2 * director)
        frames.append((D_xx, D_xy, D_yy))
    D_xx = np.stack([f[0] for f in frames])
    D_xy = np.stack([f[1] for f in frames])
    D_yy = np.stack([f[2] for f in frames])
    charges = defect_charges_over_time(D_xx, D_xy, D_yy)
    return {
        "provenance": "synthetic_proxy",
        "note": "schema-faithful planted +1/2,-1/2 defect pair; net charge expected ~0",
        "n_frames": n_frames,
        "grid": grid,
        "net_charges_per_frame": charges,
        "mean_abs_net_charge": float(np.mean(np.abs(charges))),
    }


def _real_run(dataset_root: str | None, out_dir: str) -> dict:
    """Real path: load actual active_matter trajectories via the_well and validate.

    Requires network access to HuggingFace for streaming, or a local dataset_root
    already synced via `the_well`'s own download utility.
    """
    try:
        from the_well.data import WellDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install the_well first") from exc

    ds = WellDataset(
        well_base_path=dataset_root or "hf://datasets/polymathic-ai/",
        well_dataset_name="active_matter",
        well_split_name="test",
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=False,
    )
    sample = ds[0]
    fields = sample["output_fields"] if "output_fields" in sample else sample
    # exact key names depend on the_well's tensor-dict convention; this is left explicit
    # rather than guessed blindly -- inspect `fields.keys()` on a real run and adjust.
    D_xx, D_xy, D_yy = fields["D_xx"], fields["D_xy"], fields["D_yy"]
    charges = defect_charges_over_time(np.asarray(D_xx), np.asarray(D_xy), np.asarray(D_yy))
    return {
        "provenance": "real_the_well",
        "dataset": "active_matter",
        "n_frames": len(charges),
        "net_charges_per_frame": charges,
        "mean_abs_net_charge": float(np.mean(np.abs(charges))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate topology.py against The Well active_matter")
    ap.add_argument("--dataset-root", default=None, help="Local the_well dataset root, if synced")
    ap.add_argument("--synthetic-check", action="store_true",
                     help="Run the offline schema-faithful smoke test instead of real data")
    ap.add_argument("--out", default="outputs/the_well_validation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_check:
        result = _synthetic_schema_faithful_check()
    else:
        try:
            result = _real_run(args.dataset_root, args.out)
        except Exception as exc:
            print(f"Real run failed ({exc}). This is expected if huggingface.co is not "
                  f"reachable from this environment. {NETWORK_BOUNDARY}", file=sys.stderr)
            return 2

    (out / "the_well_validation_result.json").write_text(json.dumps(result, indent=2))
    print(f"provenance={result['provenance']} n_frames={result['n_frames']} "
          f"mean_abs_net_charge={result['mean_abs_net_charge']:.4f} -> "
          f"{out/'the_well_validation_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
