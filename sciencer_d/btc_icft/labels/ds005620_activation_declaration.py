"""P17.0 — DS005620 activation declaration validator and dry-run.

Validates a human-authored DS005620 activation declaration for completeness,
provenance, and no-shortcut safeguards before any real contract activation.

This module does NOT activate any P12 contract.
It does NOT infer labels. It does NOT emit y targets.
It does NOT run P11 target-aware benchmarking.
real_contract_activation_allowed is always False.

Public API
----------
DS005620ActivationDeclaration
DS005620ActivationDeclarationValidationResult
load_activation_declaration(path)
load_activation_packet(path)
validate_activation_declaration(declaration, activation_packet=None)
build_activation_declaration_template()
build_activation_declaration_omega_event(result)
write_activation_declaration_outputs(result, out_dir)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SAFE_CLAIM = (
    "A human-authored DS005620 activation declaration was validated for "
    "completeness and no-shortcut safeguards before any real contract activation."
)

_BANNED_PHRASES: tuple[str, ...] = (
    "proves consciousness",
    "consciousness proven",
    "soul proven",
    "afterlife proven",
    "liberation detected",
    "ontology solved",
    "ultimate reality",
    "q equals self",
    "q equals soul",
    "q_abs equals suffering",
    "f_dress equals karma",
    "sedated implies no_experience",
    "unresponsive implies unconscious",
    "topology proves liberation",
    "eeg proves consciousness",
)

_FORBIDDEN_CLAIMS = [
    "No consciousness proof.",
    "No self or soul claim.",
    "No liberation or enlightenment claim.",
    "No afterlife claim.",
    "No ontology proof.",
    "No label inference.",
    "No target fabrication.",
    "No automatic contract activation.",
    "No sedated/no_experience shortcut.",
    "No unresponsive/unconscious shortcut.",
]

_VALID_DATASET_ID = "DS005620"

_VALID_LABEL_SCOPES = frozenset({"window", "file", "run", "subject", "session"})

_REQUIRED_JOIN_KEYS = [
    "dataset_id",
    "row_id",
    "source_file",
    "window_id",
    "window_start_s",
    "window_end_s",
    "sample_start",
    "sample_end",
]

_REQUIRED_DECLARATION_FIELDS = [
    "dataset_id",
    "explicit_label_column",
    "positive_values",
    "negative_values",
    "label_scope",
    "join_keys",
    "metadata_provenance",
    "semantic_justification",
    "no_shortcut_inference_confirmation",
    "reviewer_identity_or_role",
    "review_date",
    "both_classes_present_confirmation",
    "ambiguity_reviewed",
]

_NO_SHORTCUT_REQUIRED_PHRASES = (
    "no sedated-to-no_experience shortcut",
    "no unresponsive-to-unconscious shortcut",
)

_MIN_SEMANTIC_JUSTIFICATION_LEN = 40

_DEFAULT_GUARDRAILS = [
    "no_label_inference",
    "no_target_fabrication",
    "no_contract_activation_without_human_review",
    "no_sedated_to_no_experience",
    "no_unresponsive_to_unconscious",
    "no_filename_derived_labels",
    "no_topology_derived_labels",
    "no_artifact_derived_labels",
    "no_p11_gate_modification",
    "no_legacy_mt_real_change",
]


def _validate_safe_text(text: str) -> None:
    lower = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            raise ValueError(f"Banned phrase detected: {phrase!r}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DS005620ActivationDeclaration:
    dataset_id: str
    explicit_label_column: str
    positive_values: list[str]
    negative_values: list[str]
    label_scope: str
    join_keys: list[str]
    metadata_provenance: str
    semantic_justification: str
    no_shortcut_inference_confirmation: str
    reviewer_identity_or_role: str
    review_date: str
    both_classes_present_confirmation: bool
    ambiguity_reviewed: bool
    source_activation_packet: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DS005620ActivationDeclarationValidationResult:
    dataset_id: str
    declaration_valid: bool
    activation_dry_run_allowed: bool
    real_contract_activation_allowed: bool
    validation_errors: list[str]
    validation_warnings: list[str]
    normalized_declaration: dict
    activation_contract_preview: dict
    required_next_steps: list[str]
    omega_event: dict
    safe_claim: str
    forbidden_claims: list[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_activation_declaration(path: str) -> dict:
    """Load a human-authored activation declaration JSON."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Activation declaration not found: {path}"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_activation_packet(path: str) -> dict:
    """Load a P16 activation_proposal.json as cross-check context."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"P16 activation packet not found: {path}. "
            "Run prepare_ds005620_contract_activation first or omit --activation-packet."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_fields(declaration: dict) -> list[str]:
    errors: list[str] = []

    # Required fields present
    for f in _REQUIRED_DECLARATION_FIELDS:
        if f not in declaration:
            errors.append(f"missing_required_field:{f}")

    if errors:
        return errors  # Can't proceed without all fields

    # dataset_id
    if declaration["dataset_id"] != _VALID_DATASET_ID:
        errors.append(
            f"invalid_dataset_id: expected {_VALID_DATASET_ID!r}, "
            f"got {declaration['dataset_id']!r}"
        )

    # explicit_label_column
    ecol = declaration.get("explicit_label_column", "")
    if not isinstance(ecol, str) or not ecol.strip():
        errors.append("explicit_label_column must be a non-empty string")

    # positive_values / negative_values
    pos = declaration.get("positive_values", [])
    neg = declaration.get("negative_values", [])
    if not isinstance(pos, list) or len(pos) == 0:
        errors.append("positive_values must be a non-empty list")
    if not isinstance(neg, list) or len(neg) == 0:
        errors.append("negative_values must be a non-empty list")
    if isinstance(pos, list) and isinstance(neg, list) and pos and neg:
        overlap = set(pos) & set(neg)
        if overlap:
            errors.append(
                f"positive_values and negative_values must not overlap; overlap: {sorted(overlap)}"
            )

    # label_scope
    scope = declaration.get("label_scope", "")
    if scope not in _VALID_LABEL_SCOPES:
        errors.append(
            f"unsupported label_scope: {scope!r}. "
            f"Valid: {sorted(_VALID_LABEL_SCOPES)}"
        )

    # join_keys
    jk = declaration.get("join_keys", [])
    if not isinstance(jk, list):
        errors.append("join_keys must be a list")
    else:
        missing_keys = [k for k in _REQUIRED_JOIN_KEYS if k not in jk]
        if missing_keys:
            errors.append(f"join_keys missing required strict keys: {missing_keys}")

    # metadata_provenance
    prov = declaration.get("metadata_provenance", "")
    if not isinstance(prov, str) or not prov.strip():
        errors.append("metadata_provenance must be a non-empty string")
    elif prov.strip().lower() == "unknown":
        errors.append("metadata_provenance must not be 'unknown'; supply actual file path/source")

    # semantic_justification length
    sj = declaration.get("semantic_justification", "")
    if not isinstance(sj, str) or len(sj.strip()) < _MIN_SEMANTIC_JUSTIFICATION_LEN:
        errors.append(
            f"semantic_justification must be at least {_MIN_SEMANTIC_JUSTIFICATION_LEN} characters; "
            f"got {len(sj.strip() if isinstance(sj, str) else '')}"
        )

    # no_shortcut_inference_confirmation
    nsc = declaration.get("no_shortcut_inference_confirmation", "")
    if isinstance(nsc, str):
        nsc_lower = nsc.lower()
        for phrase in _NO_SHORTCUT_REQUIRED_PHRASES:
            if phrase not in nsc_lower:
                errors.append(
                    f"no_shortcut_inference_confirmation must explicitly include: {phrase!r}"
                )
    else:
        errors.append("no_shortcut_inference_confirmation must be a string")

    # both_classes_present_confirmation
    if declaration.get("both_classes_present_confirmation") is not True:
        errors.append("both_classes_present_confirmation must be true")

    # ambiguity_reviewed
    if declaration.get("ambiguity_reviewed") is not True:
        errors.append("ambiguity_reviewed must be true")

    return errors


def _cross_check_packet(declaration: dict, packet: dict) -> list[str]:
    """Cross-check declaration against P16 activation packet. Returns warnings."""
    warnings: list[str] = []

    # Guard: if packet claims activation_allowed, warn and ignore
    if packet.get("contract_activation_allowed") is True:
        warnings.append(
            "P16 packet has contract_activation_allowed=true; ignoring — "
            "P17.0 keeps real_contract_activation_allowed=false."
        )

    candidate_cols = packet.get("candidate_label_columns", [])
    unresolved = packet.get("unresolved_values", [])
    ecol = declaration.get("explicit_label_column", "")

    if candidate_cols and ecol and ecol not in candidate_cols:
        warnings.append(
            f"explicit_label_column {ecol!r} not in P16 candidate_label_columns "
            f"{candidate_cols}; verify this is the correct column."
        )

    pos = declaration.get("positive_values", [])
    neg = declaration.get("negative_values", [])
    if unresolved:
        for v in pos + neg:
            if v not in unresolved:
                warnings.append(
                    f"declared value {v!r} not in P16 unresolved_values; "
                    "confirm this value appears in local metadata."
                )

    return warnings


def validate_activation_declaration(
    declaration: dict,
    activation_packet: dict | None = None,
) -> DS005620ActivationDeclarationValidationResult:
    """Validate a human-authored activation declaration.

    real_contract_activation_allowed is always False.
    activation_dry_run_allowed is True only if declaration_valid.
    """
    errors = _validate_fields(declaration)
    warnings: list[str] = []

    if activation_packet is not None:
        pkt_warnings = _cross_check_packet(declaration, activation_packet)
        warnings.extend(pkt_warnings)

    declaration_valid = len(errors) == 0

    # Normalize declaration (only if valid enough to have required fields)
    normalized: dict = {}
    if "explicit_label_column" in declaration:
        normalized = {
            "dataset_id": declaration.get("dataset_id", ""),
            "explicit_label_column": declaration.get("explicit_label_column", ""),
            "positive_values": declaration.get("positive_values", []),
            "negative_values": declaration.get("negative_values", []),
            "label_scope": declaration.get("label_scope", "window"),
            "join_keys": declaration.get("join_keys", []),
            "metadata_provenance": declaration.get("metadata_provenance", ""),
            "semantic_justification": declaration.get("semantic_justification", ""),
            "reviewer_identity_or_role": declaration.get("reviewer_identity_or_role", ""),
            "review_date": declaration.get("review_date", ""),
            "both_classes_present_confirmation": declaration.get("both_classes_present_confirmation", False),
            "ambiguity_reviewed": declaration.get("ambiguity_reviewed", False),
            "source_activation_packet": declaration.get("source_activation_packet"),
            "notes": declaration.get("notes", []),
        }

    # Build preview (never active in source code)
    if declaration_valid:
        preview = {
            "dataset_id": declaration["dataset_id"],
            "status": "preview_human_reviewed_not_active",
            "explicit_label_column": declaration["explicit_label_column"],
            "positive_values": declaration["positive_values"],
            "negative_values": declaration["negative_values"],
            "label_scope": declaration["label_scope"],
            "join_keys": declaration["join_keys"],
            "metadata_provenance": declaration["metadata_provenance"],
            "semantic_justification": declaration["semantic_justification"],
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }
    else:
        preview = {
            "dataset_id": declaration.get("dataset_id", "DS005620"),
            "status": "preview_blocked_validation_failed",
            "explicit_label_column": None,
            "positive_values": [],
            "negative_values": [],
            "label_scope": "",
            "join_keys": [],
            "metadata_provenance": "",
            "semantic_justification": "",
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }

    next_steps = [
        "Open a separate P17.1 contract activation PR only after this declaration is valid and independently reviewed.",
        "P17.1 PR must include the validated declaration and run P12 → P13 → P11 target-aware benchmark.",
        "Do NOT merge P17.1 PR without peer review.",
    ]
    if not declaration_valid:
        next_steps.insert(0, "Fix all validation errors listed in activation_declaration_errors.json.")

    omega = build_activation_declaration_omega_event_from(
        declaration_valid, declaration.get("dataset_id", "DS005620")
    )

    return DS005620ActivationDeclarationValidationResult(
        dataset_id=declaration.get("dataset_id", "DS005620"),
        declaration_valid=declaration_valid,
        activation_dry_run_allowed=declaration_valid,
        real_contract_activation_allowed=False,
        validation_errors=errors,
        validation_warnings=warnings,
        normalized_declaration=normalized,
        activation_contract_preview=preview,
        required_next_steps=next_steps,
        omega_event=omega,
        safe_claim=_SAFE_CLAIM,
        forbidden_claims=_FORBIDDEN_CLAIMS[:],
    )


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_activation_declaration_template() -> dict:
    """Build a template with all required fields and placeholder values."""
    return {
        "dataset_id": "DS005620",
        "explicit_label_column": "<required: column name from metadata file>",
        "positive_values": ["<required: list of positive class values>"],
        "negative_values": ["<required: list of negative class values>"],
        "label_scope": "<required: window | file | run | subject | session>",
        "join_keys": _REQUIRED_JOIN_KEYS[:],
        "metadata_provenance": "<required: path/source of local metadata file, not 'unknown'>",
        "semantic_justification": (
            "<required: >= 40 chars — explain why this mapping is semantically valid "
            "and free from shortcut inference>"
        ),
        "no_shortcut_inference_confirmation": (
            "I confirm no sedated-to-no_experience shortcut and "
            "no unresponsive-to-unconscious shortcut is used in this mapping."
        ),
        "reviewer_identity_or_role": "<required: reviewer name or role>",
        "review_date": "<required: YYYY-MM-DD>",
        "both_classes_present_confirmation": False,
        "ambiguity_reviewed": False,
        "source_activation_packet": "outputs/btc_icft/ds005620_contract_activation/activation_proposal.json",
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Omega event
# ---------------------------------------------------------------------------

def build_activation_declaration_omega_event_from(
    declaration_valid: bool,
    dataset_id: str = "DS005620",
) -> dict:
    _validate_safe_text(_SAFE_CLAIM)
    payload = f"p17_0:{dataset_id}:{declaration_valid}:{_SAFE_CLAIM}"
    return {
        "event_id": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "event_type": "ds005620_activation_declaration_validation",
        "dataset_id": dataset_id,
        "declaration_valid": declaration_valid,
        "real_contract_activation_allowed": False,
        "safe_claim": _SAFE_CLAIM,
        "forbidden_claims": _FORBIDDEN_CLAIMS[:],
    }


def build_activation_declaration_omega_event(
    result: DS005620ActivationDeclarationValidationResult,
) -> dict:
    return result.omega_event


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_activation_declaration_outputs(
    result: DS005620ActivationDeclarationValidationResult,
    out_dir: str,
) -> dict[str, str]:
    """Write exactly 6 output files to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # 1. activation_declaration_template.json
    template = build_activation_declaration_template()
    p = out / "activation_declaration_template.json"
    p.write_text(json.dumps(template, indent=2), encoding="utf-8")
    outputs["activation_declaration_template"] = str(p)

    # 2. activation_declaration_validation.json
    validation_data = {
        "dataset_id": result.dataset_id,
        "declaration_valid": result.declaration_valid,
        "activation_dry_run_allowed": result.activation_dry_run_allowed,
        "real_contract_activation_allowed": result.real_contract_activation_allowed,
        "validation_errors": result.validation_errors,
        "validation_warnings": result.validation_warnings,
        "normalized_declaration": result.normalized_declaration,
        "required_next_steps": result.required_next_steps,
    }
    p = out / "activation_declaration_validation.json"
    p.write_text(json.dumps(validation_data, indent=2), encoding="utf-8")
    outputs["activation_declaration_validation"] = str(p)

    # 3. activation_contract_preview.json
    p = out / "activation_contract_preview.json"
    p.write_text(json.dumps(result.activation_contract_preview, indent=2), encoding="utf-8")
    outputs["activation_contract_preview"] = str(p)

    # 4. activation_declaration_errors.json
    errors_data = {
        "errors": result.validation_errors,
        "warnings": result.validation_warnings,
        "blocked_reason": (
            "Declaration has validation errors; fix before requesting P17.1 PR."
            if result.validation_errors
            else "Declaration is valid; proceed to P17.1 contract activation PR after independent review."
        ),
        "human_review_required": True,
    }
    p = out / "activation_declaration_errors.json"
    p.write_text(json.dumps(errors_data, indent=2), encoding="utf-8")
    outputs["activation_declaration_errors"] = str(p)

    # 5. omega_event.json
    p = out / "omega_event.json"
    p.write_text(json.dumps(result.omega_event, indent=2), encoding="utf-8")
    outputs["omega_event"] = str(p)

    # 6. report.md
    report_lines = [
        "# DS005620 Activation Declaration Validation",
        "",
        "## Stage",
        "",
        "P17.0 — DS005620 activation declaration validator and dry-run. "
        "Validates a human-authored declaration for completeness and "
        "no-shortcut safeguards before any real contract activation.",
        "",
        "## Dataset",
        "",
        f"- dataset_id: {result.dataset_id}",
        f"- declaration_valid: {result.declaration_valid}",
        f"- activation_dry_run_allowed: {result.activation_dry_run_allowed}",
        f"- real_contract_activation_allowed: {result.real_contract_activation_allowed}",
        "",
        "## Declaration Status",
        "",
    ]

    if result.declaration_valid:
        nd = result.normalized_declaration
        report_lines += [
            "Declaration is valid for dry-run preview.",
            "",
            f"- explicit_label_column: `{nd.get('explicit_label_column')}`",
            f"- positive_values: {nd.get('positive_values', [])}",
            f"- negative_values: {nd.get('negative_values', [])}",
            f"- label_scope: {nd.get('label_scope')}",
            f"- metadata_provenance: {nd.get('metadata_provenance')}",
            f"- reviewer: {nd.get('reviewer_identity_or_role')}",
            f"- review_date: {nd.get('review_date')}",
        ]
    else:
        report_lines.append("Declaration is INVALID. Fix all validation errors before proceeding.")

    report_lines += [
        "",
        "## Cross-Check Against P16 Packet",
        "",
    ]

    if result.validation_warnings:
        for w in result.validation_warnings:
            report_lines.append(f"- WARNING: {w}")
    else:
        report_lines.append("No cross-check warnings.")

    report_lines += [
        "",
        "## Activation Contract Preview",
        "",
        f"- status: `{result.activation_contract_preview.get('status')}`",
    ]

    if result.declaration_valid:
        preview = result.activation_contract_preview
        report_lines += [
            f"- explicit_label_column: `{preview.get('explicit_label_column')}`",
            f"- positive_values: {preview.get('positive_values', [])}",
            f"- negative_values: {preview.get('negative_values', [])}",
            f"- label_scope: {preview.get('label_scope')}",
            "",
            "This is a preview only. It is not active in any P12 source file.",
        ]
    else:
        report_lines.append("Preview blocked due to validation errors.")

    report_lines += [
        "",
        "## Validation Errors",
        "",
    ]
    if result.validation_errors:
        for e in result.validation_errors:
            report_lines.append(f"- ERROR: {e}")
    else:
        report_lines.append("No validation errors.")

    report_lines += [
        "",
        "## Validation Warnings",
        "",
    ]
    if result.validation_warnings:
        for w in result.validation_warnings:
            report_lines.append(f"- WARNING: {w}")
    else:
        report_lines.append("No validation warnings.")

    report_lines += [
        "",
        "## Safe Claim",
        "",
        result.safe_claim,
        "",
        "## Forbidden Claims",
        "",
    ]
    for fc in result.forbidden_claims:
        report_lines.append(f"- {fc}")

    report_lines += [
        "",
        "## Next Required Step",
        "",
    ]
    for step in result.required_next_steps:
        report_lines.append(f"- {step}")

    report_lines += [
        "",
        "## Guardrails",
        "",
        "- No label inference from file names, topology, or artifacts.",
        "- No target fabrication.",
        "- sedated does not imply no_experience.",
        "- unresponsive does not imply unconscious.",
        "- real_contract_activation_allowed is always false in this validator.",
        "- This preview does not modify any P12 source file.",
        "- Human-authored declaration requires independent review before P17.1.",
    ]

    report_text = "\n".join(report_lines) + "\n"
    _validate_safe_text(report_text)

    p = out / "report.md"
    p.write_text(report_text, encoding="utf-8")
    outputs["report"] = str(p)

    return outputs
