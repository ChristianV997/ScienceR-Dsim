"""P16 pipeline — DS005620 human-reviewed label contract activation path.

CLI for auditing local DS005620 metadata and producing an explicit contract
activation proposal and human-review packet.

Real contract activation requires a separate human-reviewed PR.
This pipeline is read-only: no labels are inferred, no targets are fabricated,
no data is downloaded, and no real contract is activated.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sciencer_d.btc_icft.labels.ds005620_contract_activation import (
    _BANNED_PHRASES,
    _SAFE_CLAIM,
    audit_ds005620_metadata,
    write_activation_outputs,
)

_REAL_ACTIVATION_BLOCKED_MSG = (
    "P16 is a read-only audit pipeline. Real contract activation requires "
    "a separate human-reviewed PR with declared explicit_label_column, "
    "positive_values, negative_values, label_scope, and join_keys. "
    "contract_activation_allowed is always false in this audit."
)


def _write_mock_metadata(ds_root: Path) -> Path:
    ds_root.mkdir(parents=True, exist_ok=True)
    events_path = ds_root / "events.tsv"
    rows = [
        {"onset": "0.0", "duration": "1.0", "trial_type": "condition_a"},
        {"onset": "1.0", "duration": "1.0", "trial_type": "condition_b"},
        {"onset": "2.0", "duration": "1.0", "trial_type": "condition_a"},
        {"onset": "3.0", "duration": "1.0", "trial_type": "condition_b"},
    ]
    with events_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["onset", "duration", "trial_type"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return events_path


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "P16: DS005620 human-reviewed label contract activation audit. "
            "Read-only. No labels inferred, no targets fabricated, no real contract activated."
        )
    )
    p.add_argument(
        "--ds-root",
        default="outputs/btc_icft/ds005620",
        help="Local DS005620 root directory to inspect.",
    )
    p.add_argument(
        "--out",
        default="outputs/btc_icft/ds005620_contract_activation",
        help="Output directory for activation proposal and review packet.",
    )
    p.add_argument(
        "--mock-fixture",
        action="store_true",
        help="Write a mock events.tsv fixture and run audit on it.",
    )
    p.add_argument(
        "--declared-label-column",
        default=None,
        help="Explicitly declare the label column (for human-reviewed use only).",
    )
    p.add_argument(
        "--declared-positive-values",
        nargs="*",
        default=None,
        help="Explicitly declare positive label values.",
    )
    p.add_argument(
        "--declared-negative-values",
        nargs="*",
        default=None,
        help="Explicitly declare negative label values.",
    )
    p.add_argument(
        "--declared-label-scope",
        default=None,
        choices=["window", "file", "subject", "session"],
        help="Explicitly declare the label scope.",
    )
    p.add_argument(
        "--declared-join-keys",
        nargs="*",
        default=None,
        help="Explicitly declare join key field names.",
    )

    a = p.parse_args(argv)

    ds_root = Path(a.ds_root)

    if a.mock_fixture:
        events_path = _write_mock_metadata(ds_root)
        print(f"[mock] Wrote mock fixture: {events_path}", file=sys.stderr)

    proposal, audit_rows = audit_ds005620_metadata(
        ds_root=str(ds_root),
        declared_label_column=a.declared_label_column,
        declared_positive_values=a.declared_positive_values,
        declared_negative_values=a.declared_negative_values,
        declared_label_scope=a.declared_label_scope,
        declared_join_keys=a.declared_join_keys,
    )

    try:
        paths = write_activation_outputs(proposal, audit_rows, a.out)
    except ValueError as exc:
        print(f"[p16] FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\n[p16] DS005620 contract activation audit complete.", file=sys.stderr)
    print(f"  metadata_file_exists: {proposal.gates.metadata_file_exists}", file=sys.stderr)
    print(f"  candidate_label_column: {proposal.candidate_label_column}", file=sys.stderr)
    print(f"  observed_values: {proposal.observed_values}", file=sys.stderr)
    print(f"  activation_blockers: {len(proposal.activation_blockers)}", file=sys.stderr)
    print(f"  contract_activation_allowed: False (always)", file=sys.stderr)
    print(f"  is_ready_for_human_review: {proposal.is_ready_for_human_review()}", file=sys.stderr)
    print(f"\n  Outputs written to: {a.out}", file=sys.stderr)
    for key, path in paths.items():
        print(f"    {key}: {path}", file=sys.stderr)

    print(f"\n  NOTE: {_REAL_ACTIVATION_BLOCKED_MSG}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
