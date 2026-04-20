from __future__ import annotations
from pathlib import Path

def find_eeg_files(root: str | Path):
    root = Path(root)
    files = []
    for ext in ("*.edf", "*.fif", "*.bdf", "*.set", "*.vhdr"):
        files.extend(sorted(root.rglob(ext)))
    return files

def find_events(root: str | Path):
    root = Path(root)
    return sorted(root.rglob("*_events.tsv"))

def dataset_manifest():
    return {
        "ds002094": {"kind": "tms_eeg", "pci": True},
        "ds005620": {"kind": "eeg_state"},
        "ds001787": {"kind": "eeg_state"},
        "ds003969": {"kind": "eeg_state"},
        "ds003816": {"kind": "eeg_state"},
    }
