"""P17.0 CLI — DS005620 activation declaration validator.

Validates a human-authored activation declaration for completeness and
no-shortcut safeguards before any real contract activation.

Does NOT activate any P12 contract.
Does NOT emit y targets.
real_contract_activation_allowed is always False.

Usage
-----
# Write template:
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \\
  --write-template --out outputs/btc_icft/ds005620_activation_declaration

# Mock valid declaration:
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \\
  --mock-valid-declaration --out outputs/btc_icft/ds005620_activation_declaration_valid

# Mock invalid declaration:
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \\
  --mock-invalid-declaration --out outputs/btc_icft/ds005620_activation_declaration_invalid

# Real/local declaration:
python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \\
  --declaration data/DS005620/ds005620_activation_declaration.json \\
  --activation-packet outputs/btc_icft/ds005620_contract_activation/activation_proposal.json \\
  --out outputs/btc_icft/ds005620_activation_declaration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sciencer_d.btc_icft.labels.ds005620_activation_declaration import (
    _REQUIRED_JOIN_KEYS,
    _SAFE_CLAIM,
    build_activation_declaration_template,
    load_activation_declaration,
    load_activation_packet,
    validate_activation_declaration,
    write_activation_declaration_outputs,
)

_NO_DECLARATION_MSG = (
    "A human-authored DS005620 activation declaration is required. "
    "Provide --declaration, --write-template, or use a mock mode."
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
    "notes": ["Mock valid declaration for P17.0 dry-run testing."],
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


def _write_template_mode(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    template = build_activation_declaration_template()
    p = out / "activation_declaration_template.json"
    p.write_text(json.dumps(template, indent=2), encoding="utf-8")
    report_lines = [
        "# DS005620 Activation Declaration Template",
        "",
        "Fill in all required fields before running P17.0 validation.",
        "",
        "## Required fields",
        "",
    ]
    for field in [
        "dataset_id", "explicit_label_column", "positive_values", "negative_values",
        "label_scope", "join_keys", "metadata_provenance", "semantic_justification",
        "no_shortcut_inference_confirmation", "reviewer_identity_or_role",
        "review_date", "both_classes_present_confirmation", "ambiguity_reviewed",
    ]:
        report_lines.append(f"- `{field}`")
    report_lines += [
        "",
        "## Next step",
        "",
        "After filling in all fields, validate with:",
        "",
        "```bash",
        "python -m sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration \\",
        "  --declaration data/DS005620/ds005620_activation_declaration.json \\",
        "  --activation-packet outputs/btc_icft/ds005620_contract_activation/activation_proposal.json \\",
        "  --out outputs/btc_icft/ds005620_activation_declaration",
        "```",
    ]
    rp = out / "template_instructions.md"
    rp.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[p17.0] Template written to: {p}", file=sys.stderr)
    print(f"[p17.0] Instructions written to: {rp}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "P17.0: DS005620 activation declaration validator. "
            "Validates human-authored declaration before any real contract activation. "
            "real_contract_activation_allowed is always false."
        )
    )
    p.add_argument(
        "--declaration",
        default=None,
        help="Path to human-authored activation declaration JSON.",
    )
    p.add_argument(
        "--activation-packet",
        default=None,
        dest="activation_packet",
        help="Path to P16 activation_proposal.json for cross-check.",
    )
    p.add_argument(
        "--out",
        default="outputs/btc_icft/ds005620_activation_declaration",
        help="Output directory.",
    )
    p.add_argument(
        "--write-template",
        action="store_true",
        default=False,
        help="Write declaration template and exit.",
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

    # Template mode
    if a.write_template:
        return _write_template_mode(a.out)

    # Resolve declaration
    if a.mock_valid_declaration:
        declaration = _MOCK_VALID_DECLARATION
        print("[p17.0] Using mock valid declaration.", file=sys.stderr)
    elif a.mock_invalid_declaration:
        declaration = _MOCK_INVALID_DECLARATION
        print("[p17.0] Using mock invalid declaration.", file=sys.stderr)
    elif a.declaration is not None:
        try:
            declaration = load_activation_declaration(a.declaration)
            print(f"[p17.0] Loaded declaration from {a.declaration}.", file=sys.stderr)
        except FileNotFoundError as exc:
            print(f"[p17.0] ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"[p17.0] ERROR: {_NO_DECLARATION_MSG}", file=sys.stderr)
        return 1

    # Resolve activation packet
    activation_packet = None
    if a.activation_packet is not None:
        try:
            activation_packet = load_activation_packet(a.activation_packet)
            print(
                f"[p17.0] Loaded P16 activation packet from {a.activation_packet}.",
                file=sys.stderr,
            )
        except FileNotFoundError as exc:
            print(f"[p17.0] ERROR: {exc}", file=sys.stderr)
            return 1

    # Validate
    result = validate_activation_declaration(declaration, activation_packet)

    # Write outputs
    try:
        paths = write_activation_declaration_outputs(result, a.out)
    except ValueError as exc:
        print(f"[p17.0] FAILED: {exc}", file=sys.stderr)
        return 1

    # Summary
    print(f"\n[p17.0] DS005620 activation declaration validation complete.", file=sys.stderr)
    print(f"  declaration_valid: {result.declaration_valid}", file=sys.stderr)
    print(f"  activation_dry_run_allowed: {result.activation_dry_run_allowed}", file=sys.stderr)
    print(f"  real_contract_activation_allowed: {result.real_contract_activation_allowed}", file=sys.stderr)
    print(f"  validation_errors: {len(result.validation_errors)}", file=sys.stderr)
    print(f"  validation_warnings: {len(result.validation_warnings)}", file=sys.stderr)

    if result.validation_errors:
        print(f"\n  Errors:", file=sys.stderr)
        for e in result.validation_errors:
            print(f"    - {e}", file=sys.stderr)

    if result.validation_warnings:
        print(f"\n  Warnings:", file=sys.stderr)
        for w in result.validation_warnings:
            print(f"    - {w}", file=sys.stderr)

    print(f"\n  Outputs written to: {a.out}", file=sys.stderr)
    for key, path in paths.items():
        print(f"    {key}: {path}", file=sys.stderr)

    print(
        "\n  NOTE: real_contract_activation_allowed is always false. "
        "Open a separate P17.1 PR after this declaration is independently reviewed.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
