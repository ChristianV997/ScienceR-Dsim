"""Generate a spec-valid, ds005620-shaped BIDS EEG dataset with SYNTHETIC signal.

This exists so the real ingestion + feature code can be exercised end-to-end in CI
and locally WITHOUT the 33GB OpenNeuro download. Signal content is synthetic and is
labelled as such by every downstream consumer. Structure/entities/metadata are real
BIDS so mne-bids discovery is genuinely tested.
"""
from __future__ import annotations
import numpy as np
import mne
from mne_bids import BIDSPath, write_raw_bids
from pathlib import Path
import shutil

def _make_raw(state: str, sfreq: float = 200.0, secs: float = 40.0, n_ch: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(sfreq * secs)
    t = np.arange(n) / sfreq
    # crude but non-trivial state contrast: 'awake' richer high-freq, 'sedated' slow-dominant
    if state == "awake":
        sig = (np.sin(2*np.pi*10*t) + 0.6*np.sin(2*np.pi*25*t)
               + 0.4*rng.standard_normal((n_ch, n)))
    else:
        sig = (1.4*np.sin(2*np.pi*2*t) + 0.2*np.sin(2*np.pi*10*t)
               + 0.4*rng.standard_normal((n_ch, n)))
        sig = np.tile(sig if sig.ndim > 1 else sig[None, :], (n_ch, 1))[:n_ch]
    sig = np.atleast_2d(sig)[:n_ch] * 1e-5  # scale to ~volts
    ch_names = [f"EEG{i:02d}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(sig, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="ignore", verbose="ERROR")
    return raw

def build(root: str, subjects=("01", "02", "03"), states=("awake", "sedated")):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for si, sub in enumerate(subjects):
        for state in states:
            raw = _make_raw(state, seed=si*10 + (0 if state=="awake" else 1))
            raw.info["line_freq"] = 60
            bp = BIDSPath(subject=sub, task=state, run="01",
                          datatype="eeg", root=root, suffix="eeg", extension=".edf")
            write_raw_bids(raw, bp, overwrite=True, allow_preload=True,
                           format="EDF", verbose="ERROR")
    # participants + description are written by write_raw_bids; return root
    return root

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/bids_synth"
    r = build(out)
    print("Synthetic BIDS root:", r)
    for p in sorted(Path(r).rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(r))
