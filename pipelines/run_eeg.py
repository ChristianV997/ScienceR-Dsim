from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False

NUMERICAL_STABILITY_EPSILON = 1e-12
MIN_SEGMENT_SAMPLES = 8

def _load_raw(path: Path):
    if not MNE_AVAILABLE:
        return None
    sfx = "".join(path.suffixes).lower()
    try:
        if sfx.endswith(".fif") or sfx.endswith(".fif.gz"):
            return mne.io.read_raw_fif(path.as_posix(), preload=True, verbose="ERROR")
        if sfx.endswith(".edf"):
            return mne.io.read_raw_edf(path.as_posix(), preload=True, verbose="ERROR")
        if sfx.endswith(".bdf"):
            return mne.io.read_raw_bdf(path.as_posix(), preload=True, verbose="ERROR")
        if sfx.endswith(".set"):
            return mne.io.read_raw_eeglab(path.as_posix(), preload=True, verbose="ERROR")
        if sfx.endswith(".vhdr"):
            return mne.io.read_raw_brainvision(path.as_posix(), preload=True, verbose="ERROR")
    except Exception:
        return None
    return None

def _preprocess(raw):
    if raw is None:
        return None
    raw = raw.copy()
    try: raw.load_data()
    except Exception: pass
    try: raw.pick_types(eeg=True, exclude=[])
    except Exception: pass
    try: raw.filter(1, 40, fir_design="firwin", verbose="ERROR")
    except Exception: pass
    try: raw.notch_filter([50, 60], verbose="ERROR")
    except Exception: pass
    try: raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    except Exception: pass
    return raw

def _compute_phase_metrics(data):
    phi = np.unwrap(np.angle(data + NUMERICAL_STABILITY_EPSILON), axis=1)
    grad = np.diff(phi, axis=1)
    Q = float(np.sum(grad) / (2 * np.pi))
    Qabs = float(np.sum(np.abs(grad)) / (2 * np.pi))
    return Q, Qabs, float(np.mean(np.abs(grad)))

def run(input_dir: str | Path, output_csv: str | Path, dataset: str, compute_pci: bool = False):
    """Run EEG preprocessing/feature extraction and save per-window metrics."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    eeg_files = []
    for ext in ("*.edf", "*.fif", "*.bdf", "*.set", "*.vhdr"):
        eeg_files.extend(sorted(input_dir.rglob(ext)))

    for f in eeg_files:
        raw = _preprocess(_load_raw(f))
        if raw is None:
            continue
        sf = float(raw.info["sfreq"])
        win = int(4 * sf)
        step = int(2 * sf)
        if raw.n_times < win:
            continue
        for s in range(0, raw.n_times - win + 1, step):
            seg = raw.get_data(start=s, stop=s + win)
            if seg.shape[1] < MIN_SEGMENT_SAMPLES:
                continue
            Q, Qabs, pg = _compute_phase_metrics(seg)
            row = {
                "dataset": dataset,
                "file": str(f),
                "start_sample": s,
                "stop_sample": s + win,
                "Q": Q,
                "Qabs": Qabs,
                "phase_grad": pg,
            }
            mean_sig = np.mean(seg, axis=0)
            fft = np.fft.rfft(mean_sig)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(mean_sig.size, d=1.0 / sf)
            low = power[(freqs >= 1) & (freqs < 8)].sum()
            high = power[(freqs >= 12) & (freqs <= 40)].sum()
            row["spectral_ratio"] = float(high / (low + NUMERICAL_STABILITY_EPSILON))
            if compute_pci:
                row["PCIst"] = np.nan
            rows.append(row)

    base_cols = [
        "dataset", "file", "start_sample", "stop_sample",
        "Q", "Qabs", "phase_grad", "spectral_ratio",
    ]
    if compute_pci:
        base_cols.append("PCIst")
    df = pd.DataFrame(rows, columns=base_cols)
    df.to_csv(output_csv, index=False)
    return df
