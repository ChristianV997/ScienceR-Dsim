"""Data adapters: route both neuroscience and quantum-field inputs to shared SU(2) fields.

Two concrete adapters over a common `DataLoader` ABC:

  * NeuroBIDSAdapter   -- reuses the already-merged data/bids_ingest.py for real BIDS EEG
                          discovery + windowed sample reads (NOT re-implemented here), then
                          extracts a spatial phase map per window via the analytic-signal
                          (Hilbert) phase across channels. provenance = "real_bids".

  * QuantumFieldAdapter-- loads an HDF5 grid (complex scalar field, or real/imag pair, or a
                          precomputed phase array) via h5py and extracts the phase-winding
                          field directly. provenance = "quantum_field".

Both return SU2Field objects, so the identical topology_engine / action_auditor code runs on
either. No cross-domain equivalence is asserted anywhere.
"""
from __future__ import annotations

import abc
from pathlib import Path
import sys
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dual_engine.su2_field_mapper import SU2Field, phase_to_su2, stack_windows_to_field  # noqa: E402


class DataLoader(abc.ABC):
    """Abstract loader: produce a list of per-window 2D phase maps + provenance."""

    provenance: str = "synthetic_proxy"

    @abc.abstractmethod
    def load_phase_windows(self, *args, **kwargs) -> list[np.ndarray]:
        """Return a list of 2D phase maps (H, W), one per time window."""

    def load_su2_field(self, *args, **kwargs) -> SU2Field:
        """Load windows and stack into a single (T, H, W) SU(2) field."""
        windows = self.load_phase_windows(*args, **kwargs)
        meta = kwargs.get("meta", {})
        return stack_windows_to_field(windows, provenance=self.provenance, meta=meta)


def _phase_map_from_channels(signal_2d: np.ndarray) -> np.ndarray:
    """Build a square spatial phase map from a (n_channels, n_samples) window.

    Uses the analytic-signal (Hilbert) instantaneous phase per channel, takes the
    phase at the window midpoint, and arranges the per-channel phases on the nearest
    square grid (zero-padded). This is a channel-agnostic spatial phase proxy; it does
    NOT perform anatomical source localization (that requires a forward model / head
    geometry not present in the generic path). Documented as such, not overclaimed.
    """
    from scipy.signal import hilbert

    analytic = hilbert(signal_2d, axis=1)
    phase = np.angle(analytic)  # (n_ch, n_samp)
    mid = phase[:, phase.shape[1] // 2]  # (n_ch,)
    n = len(mid)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.zeros(side * side, dtype=float)
    grid[:n] = mid
    return grid.reshape(side, side)


class NeuroBIDSAdapter(DataLoader):
    """Real BIDS EEG -> per-window cortical phase maps via Hilbert analytic phase.

    Reuses data.bids_ingest (already merged & tested); does not re-implement discovery
    or sample reading.
    """

    provenance = "real_bids"

    def __init__(self, bids_root: str):
        self.bids_root = bids_root

    def load_phase_windows(
        self,
        subject: Optional[str] = None,
        task: Optional[str] = None,
        n_windows: int = 6,
        window_seconds: float = 4.0,
        max_channels: int = 16,
        meta: Optional[dict] = None,
    ) -> list[np.ndarray]:
        from data.bids_ingest import discover_bids_eeg, _read_raw
        import mne

        records = discover_bids_eeg(self.bids_root)
        if subject is not None:
            records = [r for r in records if (r.subject_id or "").endswith(subject)]
        if task is not None:
            records = [r for r in records if (r.task_label or "") == task]
        if not records:
            raise FileNotFoundError(
                f"no EEG for subject={subject} task={task} under {self.bids_root}"
            )

        raw = _read_raw(records[0].path)
        raw.load_data(verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True)[:max_channels]
        sfreq = float(raw.info["sfreq"])
        win = int(window_seconds * sfreq)
        data = raw.get_data(picks=picks)  # (n_ch, n_samp) REAL samples

        windows = []
        for w in range(n_windows):
            s, e = w * win, (w + 1) * win
            if e > data.shape[1]:
                break
            windows.append(_phase_map_from_channels(data[:, s:e]))
        if not windows:
            raise ValueError("recording too short for requested windows")
        return windows

    def load_su2_field(self, subject=None, task=None, n_windows=6, window_seconds=4.0,
                       max_channels=16, meta=None) -> SU2Field:
        windows = self.load_phase_windows(
            subject=subject, task=task, n_windows=n_windows,
            window_seconds=window_seconds, max_channels=max_channels,
        )
        m = {"dataset": "bids", "subject": subject, "task": task,
             "n_windows": len(windows), **(meta or {})}
        return stack_windows_to_field(windows, provenance=self.provenance, meta=m)


class QuantumFieldAdapter(DataLoader):
    """HDF5 quantum field grid -> per-slice phase-winding maps.

    Supported HDF5 layouts (auto-detected, in order):
      1. a complex-typed dataset            -> phase = angle(field)
      2. paired real/imag datasets          -> phase = angle(re + i*im)
      3. a real 'phase' / 'theta' dataset   -> used directly
    The dataset may be 2D (single slice) or 3D (T/z, H, W). No network, no fabrication;
    if the file/keys are absent it raises rather than inventing data.
    """

    provenance = "quantum_field"

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

    def _extract_phase_stack(
        self,
        field_key: Optional[str] = None,
        real_key: Optional[str] = None,
        imag_key: Optional[str] = None,
        phase_key: Optional[str] = None,
    ) -> np.ndarray:
        import h5py

        p = Path(self.h5_path)
        if not p.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            if phase_key and phase_key in f:
                theta = np.asarray(f[phase_key][()], dtype=float)
                return theta
            if real_key and imag_key and real_key in f and imag_key in f:
                re = np.asarray(f[real_key][()], dtype=float)
                im = np.asarray(f[imag_key][()], dtype=float)
                return np.angle(re + 1j * im)
            key = field_key or (keys[0] if keys else None)
            if key is None or key not in f:
                raise KeyError(
                    f"could not find a usable dataset in {self.h5_path}; keys={keys}"
                )
            arr = f[key][()]
            if np.iscomplexobj(arr):
                return np.angle(arr)
            # real-valued single field: treat as a phase directly (already an angle)
            return np.asarray(arr, dtype=float)

    def load_phase_windows(self, field_key=None, real_key=None, imag_key=None,
                           phase_key=None, meta=None) -> list[np.ndarray]:
        theta = self._extract_phase_stack(field_key, real_key, imag_key, phase_key)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        if theta.ndim == 2:
            return [theta]
        if theta.ndim == 3:
            return [theta[i] for i in range(theta.shape[0])]
        raise ValueError(f"quantum field must be 2D or 3D, got shape {theta.shape}")

    def load_su2_field(self, field_key=None, real_key=None, imag_key=None,
                       phase_key=None, meta=None) -> SU2Field:
        windows = self.load_phase_windows(field_key, real_key, imag_key, phase_key)
        m = {"dataset": "quantum_field_h5", "path": str(self.h5_path),
             "n_slices": len(windows), **(meta or {})}
        return stack_windows_to_field(windows, provenance=self.provenance, meta=m)
