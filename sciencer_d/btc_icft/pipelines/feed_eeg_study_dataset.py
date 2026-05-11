"""CLI feeder for EEG study dataset readiness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sciencer_d.btc_icft.datasets.eeg_study_registry import (
    inspect_study_dataset,
    write_study_dataset_outputs,
)


def _generate_mock_fixture(local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    (local_root / "mock_1.csv").write_text("# channels: 4\n# sample_rate: 128\n1,2,3,4\n")
    (local_root / "mock_2.tsv").write_text("# channels: 2\n# sample_rate: 256\n1\t2\n")
    (local_root / "mock_3.txt").write_text("# channels: 1\n# sample_rate: 64\n0.1\n")


def run(dataset_id: str, local_root: str | None, out_dir: str, mock_fixture: bool = False) -> int:
    if mock_fixture:
        fixture_root = Path(out_dir) / ".mock_fixture"
        _generate_mock_fixture(fixture_root)
        local_root = str(fixture_root)

    if not local_root:
        print("Error: --local-root is required unless --mock-fixture is used.")
        return 1

    result = inspect_study_dataset(dataset_id=dataset_id, local_root=local_root)
    write_study_dataset_outputs(result, out_dir)

    if result.readiness_status == "missing_root" and not mock_fixture:
        print("Error: local root is missing or not a directory.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Feed local EEG study datasets into readiness inspection pipeline.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--local-root", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock-fixture", action="store_true")
    args = parser.parse_args()
    return run(dataset_id=args.dataset_id, local_root=args.local_root, out_dir=args.out, mock_fixture=args.mock_fixture)


if __name__ == "__main__":
    sys.exit(main())
