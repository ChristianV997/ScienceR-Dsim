"""P17.1 — DS005620 reviewed contract activation materializer.

Consumes a valid P17.0 human-authored declaration and writes a P12-compatible
active contract artifact for downstream P12 → P13 → P11 execution.

This module creates an active contract artifact ONLY from a valid declaration.
It does NOT modify source P12 contracts.
It does NOT infer labels. It does NOT emit y targets.
It does NOT run P11 benchmarking.

Public API
----------
DS005620ReviewedContractActivation
DS005620ReviewedContractActivationResult
load_validated_declaration(path)
load_declaration_validation(path)
materialize_reviewed_contract(declaration, validation=None)
build_p12_external_contract(result)
build_p18_handoff(result)
build_reviewed_contract_omega_event(result)
write_reviewed_contract_outputs(result, out_dir)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sciencer_d.btc_icft.labels.ds005620_activation_declaration import (
    _REQUIRED_JOIN_KEYS,
    _NO_SHORTCUT_REQUIRED_PHRASES,
    _VALID_DATASET_ID,
    _VALID_LABEL_SCOPES,
    validate_activation_declaration,
)

_SAFE_CLAIM = (
    "A valid human-authored DS005620 declaration was materialized into a "
    "reviewed external contract artifact without inferring labels or targets."
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
    "No source contract modified.",
    "No P11 run executed.",
]

_DEFAULT_GUARDRAILS = [
    "no_label_inference",
    "no_target_fabrication",
    "no_source_contract_modification",
    "no_automatic_benchmark_run",
    "no_sedated_to_no_experience",
    "no_unresponsive_to_unconscious",
    "no_filename_derived_labels",
    "no_topology_derived_labels",
    "no_artifact_derived_labels",
    "no_p11_gate_modification",
    "no_legacy_mt_real_change",
]

_P18_REQUIRED_INPUTS = [
    "real/local DS005620 metadata file (events.tsv or equivalent)",
    "readable DS005620 signal windows or MNE extraction outputs (P19.1)",
    "P12 alignment output (label_alignment.csv)",
    "P13 labeled features (features_m_signal_labeled.csv)",
    "P10 topology features (features_t_signal.csv)",
    "P11 target-aware benchmark output (metrics_signal_mt.json)",
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
class DS005620ReviewedContractActivation:
    dataset_id: str
    status: str
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
    activation_source: str
    activation_allowed: bool
    guardrails: list[str] = field(default_factory=list)


@dataclass
class DS005620ReviewedContractActivationResult:
    dataset_id: str
    reviewed_contract_valid: bool
    activation_allowed: bool
    p12_external_contract: dict
    p18_handoff: dict
    activation_errors: list[str]
    activation_warnings: list[str]
    omega_event: dict
    safe_claim: str
    forbidden_claims: list[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_validated_declaration(path: str) -> dict:
    """Load a human-authored activation declaration JSON."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Activation declaration not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_declaration_validation(path: str) -> dict:
    """Load a P17.0 activation_declaration_validation.json."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"P17.0 validation output not found: {path}. "
            "Run validate_ds005620_activation_declaration first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Activation rules
# ---------------------------------------------------------------------------

def _extract_declaration_from_validation(validation: dict) -> tuple[dict, list[str]]:
    """Extract normalized declaration from P17.0 output."""
    warnings: list[str] = []
    if validation.get("real_contract_activation_allowed") is True:
        warnings.append(
            "P17.0 validation has real_contract_activation_allowed=true; "
            "ignoring — P17.1 maintains full validation independently."
        )
    nd = validation.get("normalized_declaration", {})
    if not nd:
        return {}, warnings
    return nd, warnings


def _validate_declaration_for_materialization(declaration: dict) -> list[str]:
    """Run P17.0-equivalent validation for materialization guard."""
    validation_result = validate_activation_declaration(declaration)
    return validation_result.validation_errors


def _build_activation_errors(declaration: dict) -> list[str]:
    return _validate_declaration_for_materialization(declaration)


# ---------------------------------------------------------------------------
# Core materializer
# ---------------------------------------------------------------------------

def materialize_reviewed_contract(
    declaration: dict,
    validation: dict | None = None,
) -> DS005620ReviewedContractActivationResult:
    """Materialize a reviewed external contract from a valid declaration.

    activation_allowed is True in the output artifact only if all validation
    checks pass. No source P12 contracts are modified. No P11 is run.
    No labels are inferred. No y targets are emitted.
    """
    warnings: list[str] = []

    # If validation JSON provided, cross-check and extract declaration if needed
    if validation is not None:
        extracted, val_warnings = _extract_declaration_from_validation(validation)
        warnings.extend(val_warnings)

        # If declaration not separately provided, use normalized from validation
        if not declaration and extracted:
            declaration = extracted
        elif extracted:
            # Check consistency
            if extracted.get("dataset_id") != declaration.get("dataset_id"):
                warnings.append(
                    "Declaration dataset_id does not match validation normalized_declaration; "
                    "using the raw declaration for materialization."
                )

        # Propagate validation warnings
        for w in validation.get("validation_warnings", []):
            if w not in warnings:
                warnings.append(w)

    # Validate declaration
    errors = _build_activation_errors(declaration)
    reviewed_contract_valid = len(errors) == 0
    activation_allowed = reviewed_contract_valid

    dataset_id = declaration.get("dataset_id", "DS005620")

    # Build contract objects
    if reviewed_contract_valid:
        contract = DS005620ReviewedContractActivation(
            dataset_id=dataset_id,
            status="active_reviewed_external_contract",
            explicit_label_column=declaration["explicit_label_column"],
            positive_values=declaration["positive_values"],
            negative_values=declaration["negative_values"],
            label_scope=declaration["label_scope"],
            join_keys=declaration["join_keys"],
            metadata_provenance=declaration["metadata_provenance"],
            semantic_justification=declaration["semantic_justification"],
            no_shortcut_inference_confirmation=declaration["no_shortcut_inference_confirmation"],
            reviewer_identity_or_role=declaration["reviewer_identity_or_role"],
            review_date=declaration["review_date"],
            activation_source="p17_1_reviewed_materializer",
            activation_allowed=True,
            guardrails=_DEFAULT_GUARDRAILS[:],
        )
        contract_dict = asdict(contract)

        p12 = build_p12_external_contract_from(contract)
        p18 = build_p18_handoff_from(contract, valid=True, warnings=warnings)
    else:
        contract_dict = {
            "dataset_id": dataset_id,
            "status": "blocked_invalid_declaration",
            "explicit_label_column": None,
            "positive_values": [],
            "negative_values": [],
            "label_scope": "",
            "join_keys": [],
            "metadata_provenance": "",
            "semantic_justification": "",
            "no_shortcut_inference_confirmation": "",
            "reviewer_identity_or_role": "",
            "review_date": "",
            "activation_source": "p17_1_reviewed_materializer",
            "activation_allowed": False,
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }
        p12 = {
            "dataset_id": dataset_id,
            "contract_status": "blocked_invalid_declaration",
            "explicit_label_column": None,
            "positive_values": [],
            "negative_values": [],
            "label_scope": "",
            "join_keys": [],
            "metadata_provenance": "",
            "activation_provenance": "p17_1_reviewed_materializer",
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }
        p18 = build_p18_handoff_from(None, valid=False, warnings=warnings, errors=errors)

    omega = build_reviewed_contract_omega_event_from(reviewed_contract_valid, dataset_id)

    return DS005620ReviewedContractActivationResult(
        dataset_id=dataset_id,
        reviewed_contract_valid=reviewed_contract_valid,
        activation_allowed=activation_allowed,
        p12_external_contract=p12,
        p18_handoff=p18,
        activation_errors=errors,
        activation_warnings=warnings,
        omega_event=omega,
        safe_claim=_SAFE_CLAIM,
        forbidden_claims=_FORBIDDEN_CLAIMS[:],
    )


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------

def build_p12_external_contract_from(
    contract: DS005620ReviewedContractActivation,
) -> dict:
    return {
        "dataset_id": contract.dataset_id,
        "contract_status": contract.status,
        "explicit_label_column": contract.explicit_label_column,
        "positive_values": contract.positive_values,
        "negative_values": contract.negative_values,
        "label_scope": contract.label_scope,
        "join_keys": contract.join_keys,
        "metadata_provenance": contract.metadata_provenance,
        "activation_provenance": "p17_1_reviewed_materializer",
        "guardrails": contract.guardrails,
    }


def build_p18_handoff_from(
    contract: DS005620ReviewedContractActivation | None,
    valid: bool,
    warnings: list[str],
    errors: list[str] | None = None,
) -> dict:
    blockers: list[str] = []
    if not valid:
        blockers.append("Declaration validation failed; fix errors before running P18.")
        if errors:
            blockers.extend(errors)

    recommended = []
    if valid:
        recommended = [
            "python -m sciencer_d.btc_icft.pipelines.align_eeg_labels "
            "--dataset-id DS005620 "
            "--external-contract outputs/btc_icft/ds005620_reviewed_contract/p12_external_contract.json "
            "--out outputs/btc_icft/eeg_labels/DS005620_reviewed",
        ]

    return {
        "dataset_id": "DS005620",
        "ready_for_p12_alignment": valid,
        "ready_for_p13_target_injection": valid,
        "ready_for_p11_target_aware_benchmark": valid,
        "required_inputs_for_p18": _P18_REQUIRED_INPUTS[:],
        "recommended_commands": recommended,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_p12_external_contract(
    result: DS005620ReviewedContractActivationResult,
) -> dict:
    return result.p12_external_contract


def build_p18_handoff(
    result: DS005620ReviewedContractActivationResult,
) -> dict:
    return result.p18_handoff


def build_reviewed_contract_omega_event_from(
    reviewed_contract_valid: bool,
    dataset_id: str = "DS005620",
) -> dict:
    _validate_safe_text(_SAFE_CLAIM)
    payload = f"p17_1:{dataset_id}:{reviewed_contract_valid}:{_SAFE_CLAIM}"
    return {
        "event_id": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "event_type": "ds005620_reviewed_contract_materialization",
        "dataset_id": dataset_id,
        "reviewed_contract_valid": reviewed_contract_valid,
        "activation_allowed_in_artifact": reviewed_contract_valid,
        "source_contracts_modified": False,
        "p11_run": False,
        "safe_claim": _SAFE_CLAIM,
        "forbidden_claims": _FORBIDDEN_CLAIMS[:],
    }


def build_reviewed_contract_omega_event(
    result: DS005620ReviewedContractActivationResult,
) -> dict:
    return result.omega_event


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_reviewed_contract_outputs(
    result: DS005620ReviewedContractActivationResult,
    out_dir: str,
) -> dict[str, str]:
    """Write exactly 6 output files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # 1. reviewed_contract.json
    if result.reviewed_contract_valid:
        reviewed_data = {
            "dataset_id": result.dataset_id,
            "status": "active_reviewed_external_contract",
            **{k: v for k, v in result.p12_external_contract.items()
               if k not in ("dataset_id", "contract_status")},
            "contract_status": "active_reviewed_external_contract",
            "activation_allowed": True,
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }
    else:
        reviewed_data = {
            "dataset_id": result.dataset_id,
            "status": "blocked_invalid_declaration",
            "activation_allowed": False,
            "activation_errors": result.activation_errors,
            "guardrails": _DEFAULT_GUARDRAILS[:],
        }
    p = out / "reviewed_contract.json"
    p.write_text(json.dumps(reviewed_data, indent=2), encoding="utf-8")
    outputs["reviewed_contract"] = str(p)

    # 2. p12_external_contract.json
    p = out / "p12_external_contract.json"
    p.write_text(json.dumps(result.p12_external_contract, indent=2), encoding="utf-8")
    outputs["p12_external_contract"] = str(p)

    # 3. p18_handoff.json
    p = out / "p18_handoff.json"
    p.write_text(json.dumps(result.p18_handoff, indent=2), encoding="utf-8")
    outputs["p18_handoff"] = str(p)

    # 4. reviewed_activation_report.json
    report_data = {
        "reviewed_contract_valid": result.reviewed_contract_valid,
        "activation_allowed": result.activation_allowed,
        "activation_errors": result.activation_errors,
        "activation_warnings": result.activation_warnings,
        "no_label_inference": True,
        "no_target_fabrication": True,
        "no_source_contract_modified": True,
        "no_p11_run": True,
    }
    p = out / "reviewed_activation_report.json"
    p.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    outputs["reviewed_activation_report"] = str(p)

    # 5. omega_event.json
    p = out / "omega_event.json"
    p.write_text(json.dumps(result.omega_event, indent=2), encoding="utf-8")
    outputs["omega_event"] = str(p)

    # 6. report.md
    prc = result.p12_external_contract
    report_lines = [
        "# DS005620 Reviewed External Contract Activation",
        "",
        "## Stage",
        "",
        "P17.1 — DS005620 reviewed contract materializer. "
        "Consumes a valid human-authored declaration and produces a "
        "reviewed external contract artifact for downstream P12 alignment. "
        "No source contract modified. No labels inferred. No targets fabricated.",
        "",
        "## Dataset",
        "",
        f"- dataset_id: {result.dataset_id}",
        f"- reviewed_contract_valid: {result.reviewed_contract_valid}",
        f"- activation_allowed: {result.activation_allowed}",
        "",
        "## Reviewed Declaration",
        "",
    ]
    if result.reviewed_contract_valid:
        report_lines += [
            f"- explicit_label_column: `{prc.get('explicit_label_column')}`",
            f"- positive_values: {prc.get('positive_values', [])}",
            f"- negative_values: {prc.get('negative_values', [])}",
            f"- label_scope: {prc.get('label_scope')}",
            f"- metadata_provenance: {prc.get('metadata_provenance')}",
            f"- activation_provenance: {prc.get('activation_provenance')}",
        ]
    else:
        report_lines.append(
            "Declaration is INVALID. Reviewed external contract was not materialized."
        )
        report_lines.append("")
        for e in result.activation_errors:
            report_lines.append(f"- ERROR: {e}")

    report_lines += [
        "",
        "## External Contract Artifact",
        "",
        f"- status: `{prc.get('contract_status', 'blocked_invalid_declaration')}`",
        "- This artifact is a reviewed external contract. "
        "It is NOT written into any P12 source file.",
        "- P12 alignment should load this artifact via --external-contract flag.",
        "",
        "## P12 Compatibility",
        "",
        "The p12_external_contract.json uses P12-compatible field names:",
        "dataset_id, contract_status, explicit_label_column, positive_values,",
        "negative_values, label_scope, join_keys, metadata_provenance,",
        "activation_provenance, guardrails.",
        "",
        "## P18 Handoff",
        "",
        f"- ready_for_p12_alignment: {result.p18_handoff.get('ready_for_p12_alignment')}",
        f"- ready_for_p13_target_injection: {result.p18_handoff.get('ready_for_p13_target_injection')}",
        f"- ready_for_p11_target_aware_benchmark: {result.p18_handoff.get('ready_for_p11_target_aware_benchmark')}",
        "",
        "### Required P18 inputs",
        "",
    ]
    for inp in result.p18_handoff.get("required_inputs_for_p18", []):
        report_lines.append(f"- {inp}")
    if result.p18_handoff.get("blockers"):
        report_lines += ["", "### P18 blockers", ""]
        for b in result.p18_handoff["blockers"]:
            report_lines.append(f"- {b}")

    report_lines += [
        "",
        "## Guardrails",
        "",
        "- No label inference from file names, topology, or artifacts.",
        "- No target fabrication.",
        "- No source contract modified (reviewed external contract artifact only).",
        "- No P11 run executed.",
        "- sedated does not imply no_experience.",
        "- unresponsive does not imply unconscious.",
        "- without inferring labels or targets.",
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
        "Run P18 only after local DS005620 metadata and signal extraction "
        "artifacts are available and the reviewed external contract has been peer checked.",
        "",
        "1. Verify p12_external_contract.json with peer reviewer.",
        "2. Run P12 alignment with --external-contract flag.",
        "3. Run P13 target injection.",
        "4. Run P11 target-aware benchmark.",
        "5. Open P18 PR with all benchmark artifacts.",
    ]

    report_text = "\n".join(report_lines) + "\n"
    _validate_safe_text(report_text)

    p = out / "report.md"
    p.write_text(report_text, encoding="utf-8")
    outputs["report"] = str(p)

    return outputs
