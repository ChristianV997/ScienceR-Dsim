"""P17.1 CLI — DS005620 reviewed contract activation materializer.

Consumes a valid P17.0 human-authored declaration and materializes a
P12-compatible reviewed external contract artifact for downstream P18 use.

Does NOT modify any P12 source file.
Does NOT infer labels or emit y targets.
Does NOT run P11 benchmarking.
activation_allowed is True in output artifact ONLY when declaration is valid.

Usage
-----
# Mock valid declaration:
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \\
  --mock-valid-declaration --out outputs/btc_icft/ds005620_reviewed_contract

# Mock invalid declaration:
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \\
  --mock-invalid-declaration --out outputs/btc_icft/ds005620_reviewed_contract_invalid

# Real/local declaration (with optional P17.0 validation cross-check):
python -m sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract \\
  --declaration data/DS005620/ds005620_activation_declaration.json \\
  --validation outputs/btc_icft/ds005620_activation_declaration/activation_declaration_validation.json \\
  --out outputs/btc_icft/ds005620_reviewed_contract
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciencer_d.btc_icft.labels.ds005620_activation_declaration import _REQUIRED_JOIN_KEYS
from sciencer_d.btc_icft.labels.ds005620_reviewed_contract_activation import (
    load_declaration_validation,
    load_validated_declaration,
    materialize_reviewed_contract,
    write_reviewed_contract_outputs,
)

_NO_DECLARATION_MSG = (
    "A valid DS005620 activation declaration is required. "
    "Provide --declaration/--validation or use --mock-valid-declaration."
)

_MOCK_VALID_DECLARATION = {
    "dataset_id": "DS005620",
    "explicit_label_column": "trial_type",
    "positive_values": ["focus"],
    "negative_values": ["mind_wandering"],
    "label_scope": "window",
    "join_keys": _REQUIRED_JOIN_KEYS[:],
    "metadata_provenance": "data/DS005620/events.tsv (local BIDS events sidecar)",
    "semantic_justification": (
        "The trial_type column contains experimenter-assigned condition labels "
        "from the original study protocol. 'focus' denotes sustained attention "
        "blocks and 'mind_wandering' denotes rest/distracted blocks as defined "
        "in the study design. No inference from EEG signal, topology, or file "
        "names is performed. The mapping is based solely on the study protocol."
    ),
    "no_shortcut_inference_confirmation": (
        "I confirm no sedated-to-no_experience shortcut and "
        "no unresponsive-to-unconscious shortcut is used in this mapping. "
        "Labels are derived only from the declared metadata column."
    ),
    "reviewer_identity_or_role": "lead_data_curator",
    "review_date": "2026-05-12",
    "both_classes_present_confirmation": True,
    "ambiguity_reviewed": True,
    "source_activation_packet": (
        "outputs/btc_icft/ds005620_contract_activation/activation_proposal.json"
    ),
    "notes": ["Mock valid declaration for P17.1 dry-run testing."],
}

_MOCK_INVALID_DECLARATION = {
    "dataset_id": "DS005620",
    "explicit_label_column": "",
    "positive_values": [],
    "negative_values": [],
    "label_scope": "invalid_scope",
    "join_keys": ["dataset_id"],
    "metadata_provenance": "unknown",
    "semantic_justification": "short",
    "no_shortcut_inference_confirmation": "not complete",
    "reviewer_identity_or_role": "mock_reviewer",
    "review_date": "2026-05-12",
    "both_classes_present_confirmation": False,
    "ambiguity_reviewed": False,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "P17.1: DS005620 reviewed contract activation materializer. "
            "Materializes a P12-compatible reviewed external contract from a "
            "valid human-authored declaration. "
            "Does not modify source P12 contracts. "
            "activation_allowed is True in output artifact only for valid declarations."
        )
    )
    p.add_argument(
        "--declaration",
        default=None,
        help="Path to human-authored activation declaration JSON.",
    )
    p.add_argument(
        "--validation",
        default=None,
        help="Path to P17.0 activation_declaration_validation.json for cross-check.",
    )
    p.add_argument(
        "--out",
        default="outputs/btc_icft/ds005620_reviewed_contract",
        help="Output directory.",
    )
    p.add_argument(
        "--mock-valid-declaration",
        action="store_true",
        default=False,
        help="Use deterministic valid mock declaration.",
    )
    p.add_argument(
        "--mock-invalid-declaration",
        action="store_true",
        default=False,
        help="Use deterministic invalid mock declaration.",
    )

    a = p.parse_args(argv)

    # Resolve declaration
    if a.mock_valid_declaration:
        declaration = _MOCK_VALID_DECLARATION
        validation = None
        print("[p17.1] Using mock valid declaration.", file=sys.stderr)
    elif a.mock_invalid_declaration:
        declaration = _MOCK_INVALID_DECLARATION
        validation = None
        print("[p17.1] Using mock invalid declaration.", file=sys.stderr)
    elif a.declaration is not None:
        try:
            declaration = load_validated_declaration(a.declaration)
            print(f"[p17.1] Loaded declaration from {a.declaration}.", file=sys.stderr)
        except FileNotFoundError as exc:
            print(f"[p17.1] ERROR: {exc}", file=sys.stderr)
            return 1
        validation = None
        if a.validation is not None:
            try:
                validation = load_declaration_validation(a.validation)
                print(
                    f"[p17.1] Loaded P17.0 validation from {a.validation}.",
                    file=sys.stderr,
                )
            except FileNotFoundError as exc:
                print(f"[p17.1] ERROR: {exc}", file=sys.stderr)
                return 1
    else:
        print(f"[p17.1] ERROR: {_NO_DECLARATION_MSG}", file=sys.stderr)
        return 1

    # Materialize
    result = materialize_reviewed_contract(declaration, validation)

    # Write outputs
    try:
        paths = write_reviewed_contract_outputs(result, a.out)
    except ValueError as exc:
        print(f"[p17.1] FAILED: {exc}", file=sys.stderr)
        return 1

    # Summary
    print(
        "\n[p17.1] DS005620 reviewed contract materialization complete.",
        file=sys.stderr,
    )
    print(
        f"  reviewed_contract_valid: {result.reviewed_contract_valid}",
        file=sys.stderr,
    )
    print(f"  activation_allowed: {result.activation_allowed}", file=sys.stderr)
    print(
        f"  activation_errors: {len(result.activation_errors)}",
        file=sys.stderr,
    )
    print(
        f"  activation_warnings: {len(result.activation_warnings)}",
        file=sys.stderr,
    )

    if result.activation_errors:
        print("\n  Errors:", file=sys.stderr)
        for e in result.activation_errors:
            print(f"    - {e}", file=sys.stderr)

    if result.activation_warnings:
        print("\n  Warnings:", file=sys.stderr)
        for w in result.activation_warnings:
            print(f"    - {w}", file=sys.stderr)

    print(f"\n  Outputs written to: {a.out}", file=sys.stderr)
    for key, path in paths.items():
        print(f"    {key}: {path}", file=sys.stderr)

    p18 = result.p18_handoff
    print(
        f"\n  P18 readiness:",
        file=sys.stderr,
    )
    print(
        f"    ready_for_p12_alignment: {p18.get('ready_for_p12_alignment')}",
        file=sys.stderr,
    )
    print(
        f"    ready_for_p13_target_injection: {p18.get('ready_for_p13_target_injection')}",
        file=sys.stderr,
    )
    print(
        f"    ready_for_p11_target_aware_benchmark: {p18.get('ready_for_p11_target_aware_benchmark')}",
        file=sys.stderr,
    )

    print(
        "\n  NOTE: No source P12 contract was modified. "
        "The reviewed external contract is an output artifact only. "
        "Open a P18 PR after peer review of p12_external_contract.json.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
