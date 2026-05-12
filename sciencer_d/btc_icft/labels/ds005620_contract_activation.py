"""P16 — DS005620 human-reviewed label contract activation path.

Inspects local DS005620 metadata files and produces an explicit contract
activation proposal and human-review packet. Does NOT activate any real
label contract automatically. Real contract activation requires a separate
human-reviewed PR with all required fields declared.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_SAFE_CLAIM = (
    "DS005620 local metadata was audited for explicit human-reviewed "
    "label-contract activation readiness without inferring labels or targets."
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
    "sedated -> no_experience",
    "unresponsive -> unconscious",
)

_CANDIDATE_LABEL_COLUMNS: tuple[str, ...] = (
    "trial_type",
    "condition",
    "state",
    "label",
    "task",
    "category",
    "class",
    "group",
    "annotation",
    "behavior",
    "report_label",
    "state_label",
)

_AMBIGUOUS_VALUES: frozenset[str] = frozenset({
    "n/a",
    "na",
    "nan",
    "none",
    "unknown",
    "undefined",
    "",
    "?",
    "other",
})

_REQUIRED_CONTRACT_FIELDS: tuple[str, ...] = (
    "explicit_label_column",
    "positive_values",
    "negative_values",
    "label_scope",
    "join_keys",
)

DS005620_CANDIDATE_METADATA_FILES: tuple[str, ...] = (
    "events.tsv",
    "events.csv",
    "*_events.tsv",
    "*_events.csv",
    "participants.tsv",
    "participants.csv",
    "phenotype/*.tsv",
    "phenotype/*.csv",
)


@dataclass
class MetadataValueAuditRow:
    column: str
    value: str
    count: int
    is_ambiguous: bool
    candidate_positive: bool
    candidate_negative: bool


@dataclass
class ActivationGates:
    metadata_file_exists: bool = False
    explicit_label_column_declared: bool = False
    positive_values_declared: bool = False
    negative_values_declared: bool = False
    label_scope_declared: bool = False
    join_keys_declared: bool = False
    both_classes_present: bool = False
    ambiguous_values_rejected: bool = False
    human_review_required: bool = True
    contract_activation_allowed: bool = False

    def all_structural_gates_passed(self) -> bool:
        return (
            self.metadata_file_exists
            and self.explicit_label_column_declared
            and self.positive_values_declared
            and self.negative_values_declared
            and self.label_scope_declared
            and self.join_keys_declared
            and self.both_classes_present
            and self.ambiguous_values_rejected
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActivationProposal:
    dataset_id: str
    gates: ActivationGates
    candidate_metadata_file: Optional[str]
    candidate_label_column: Optional[str]
    observed_values: list[str]
    ambiguous_values_found: list[str]
    activation_blockers: list[str]
    suggested_positive_values: list[str]
    suggested_negative_values: list[str]
    notes: list[str]
    safe_claim: str = _SAFE_CLAIM

    def is_ready_for_human_review(self) -> bool:
        return self.gates.metadata_file_exists and len(self.activation_blockers) == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = self.gates.to_dict()
        d["is_ready_for_human_review"] = self.is_ready_for_human_review()
        return d


@dataclass
class HumanReviewPacket:
    dataset_id: str
    proposal_summary: str
    required_declarations: list[str]
    activation_checklist: list[dict]
    forbidden_shortcuts: list[str]
    reviewer_instructions: str
    safe_claim: str = _SAFE_CLAIM

    def to_dict(self) -> dict:
        return asdict(self)


def _discover_metadata_file(ds_root: Path) -> Optional[Path]:
    patterns = [
        "events.tsv", "events.csv",
        "*_events.tsv", "*_events.csv",
        "**/*_events.tsv", "**/*_events.csv",
        "participants.tsv", "participants.csv",
    ]
    for pat in patterns:
        matches = sorted(ds_root.rglob(pat.lstrip("**/").lstrip("*")))
        if matches:
            return matches[0]
    return None


def _read_tabular(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    return list(reader)


def _detect_label_column(rows: list[dict]) -> Optional[str]:
    if not rows:
        return None
    columns = list(rows[0].keys())
    for col in _CANDIDATE_LABEL_COLUMNS:
        if col in columns:
            return col
    for col in columns:
        col_lower = col.lower()
        for keyword in ("type", "label", "cond", "state", "class", "task", "group"):
            if keyword in col_lower:
                return col
    return None


def _collect_value_counts(rows: list[dict], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        val = row.get(column, "").strip().lower()
        counts[val] = counts.get(val, 0) + 1
    return counts


def _build_audit_rows(value_counts: dict[str, int]) -> list[MetadataValueAuditRow]:
    audit = []
    values = sorted(value_counts.keys())
    for val in values:
        audit.append(MetadataValueAuditRow(
            column="detected_label_column",
            value=val,
            count=value_counts[val],
            is_ambiguous=val in _AMBIGUOUS_VALUES,
            candidate_positive=val not in _AMBIGUOUS_VALUES,
            candidate_negative=val not in _AMBIGUOUS_VALUES,
        ))
    return audit


def _build_activation_blockers(
    metadata_path: Optional[Path],
    label_column: Optional[str],
    value_counts: dict[str, int],
) -> list[str]:
    blockers = []
    if metadata_path is None:
        blockers.append("No metadata file found in DS005620 local root.")
        return blockers
    if label_column is None:
        blockers.append("No candidate label column detected in metadata file.")
        return blockers
    clean_values = {v for v in value_counts if v not in _AMBIGUOUS_VALUES}
    if len(clean_values) < 2:
        blockers.append(
            f"Insufficient distinct label values (found {len(clean_values)}; "
            "need at least 2 non-ambiguous values for binary contract)."
        )
    ambiguous = [v for v in value_counts if v in _AMBIGUOUS_VALUES and value_counts[v] > 0]
    if ambiguous:
        blockers.append(
            f"Ambiguous values present that must be explicitly excluded: {ambiguous}. "
            "Declare which values map to positive/negative and exclude ambiguous ones."
        )
    blockers.append(
        "Contract activation requires explicit human-reviewed PR with declared "
        "explicit_label_column, positive_values, negative_values, label_scope, and join_keys."
    )
    return blockers


def audit_ds005620_metadata(
    ds_root: str,
    declared_label_column: Optional[str] = None,
    declared_positive_values: Optional[list[str]] = None,
    declared_negative_values: Optional[list[str]] = None,
    declared_label_scope: Optional[str] = None,
    declared_join_keys: Optional[list[str]] = None,
) -> tuple[ActivationProposal, list[MetadataValueAuditRow]]:
    root = Path(ds_root)
    metadata_path = _discover_metadata_file(root)

    rows: list[dict] = []
    label_column: Optional[str] = None
    value_counts: dict[str, int] = {}
    audit_rows: list[MetadataValueAuditRow] = []

    if metadata_path is not None and metadata_path.is_file():
        try:
            rows = _read_tabular(metadata_path)
        except Exception:
            rows = []
        label_column = declared_label_column or _detect_label_column(rows)
        if label_column and rows:
            value_counts = _collect_value_counts(rows, label_column)
            audit_rows = _build_audit_rows(value_counts)

    clean_values = sorted(v for v in value_counts if v not in _AMBIGUOUS_VALUES)
    ambiguous_found = sorted(v for v in value_counts if v in _AMBIGUOUS_VALUES)

    pos_vals = declared_positive_values or []
    neg_vals = declared_negative_values or []

    gates = ActivationGates(
        metadata_file_exists=metadata_path is not None and len(rows) > 0,
        explicit_label_column_declared=declared_label_column is not None,
        positive_values_declared=len(pos_vals) > 0,
        negative_values_declared=len(neg_vals) > 0,
        label_scope_declared=declared_label_scope is not None,
        join_keys_declared=declared_join_keys is not None and len(declared_join_keys) > 0,
        both_classes_present=(
            len([v for v in clean_values if v in [pv.lower() for pv in pos_vals]]) > 0
            and len([v for v in clean_values if v in [nv.lower() for nv in neg_vals]]) > 0
        ) if (pos_vals and neg_vals and clean_values) else False,
        ambiguous_values_rejected=len(ambiguous_found) == 0,
        human_review_required=True,
        contract_activation_allowed=False,
    )

    blockers = _build_activation_blockers(metadata_path, label_column, value_counts)

    suggested_pos = clean_values[:1] if clean_values else []
    suggested_neg = clean_values[1:2] if len(clean_values) > 1 else []

    proposal = ActivationProposal(
        dataset_id="DS005620",
        gates=gates,
        candidate_metadata_file=str(metadata_path) if metadata_path else None,
        candidate_label_column=label_column,
        observed_values=clean_values,
        ambiguous_values_found=ambiguous_found,
        activation_blockers=blockers,
        suggested_positive_values=suggested_pos,
        suggested_negative_values=suggested_neg,
        notes=[
            "Inspection is read-only. No labels were inferred or targets fabricated.",
            "Real contract activation requires separate human-reviewed PR.",
            "Declared positive/negative values must be semantically validated by a human reviewer.",
        ],
    )
    return proposal, audit_rows


def build_human_review_packet(proposal: ActivationProposal) -> HumanReviewPacket:
    checklist = [
        {"item": "Verify metadata_file_exists == true", "status": proposal.gates.metadata_file_exists},
        {"item": "Declare explicit_label_column", "status": proposal.gates.explicit_label_column_declared},
        {"item": "Declare positive_values (non-ambiguous)", "status": proposal.gates.positive_values_declared},
        {"item": "Declare negative_values (non-ambiguous)", "status": proposal.gates.negative_values_declared},
        {"item": "Declare label_scope (window/file/subject/session)", "status": proposal.gates.label_scope_declared},
        {"item": "Declare join_keys (composite key fields)", "status": proposal.gates.join_keys_declared},
        {"item": "Confirm both classes present in local metadata", "status": proposal.gates.both_classes_present},
        {"item": "Confirm ambiguous values explicitly excluded", "status": proposal.gates.ambiguous_values_rejected},
        {"item": "Human review completed", "status": False},
        {"item": "Contract activation PR created with all required declarations", "status": False},
    ]
    return HumanReviewPacket(
        dataset_id=proposal.dataset_id,
        proposal_summary=(
            f"DS005620 metadata audit found "
            f"{'a candidate metadata file' if proposal.candidate_metadata_file else 'no metadata file'}. "
            f"Candidate label column: {proposal.candidate_label_column or 'none detected'}. "
            f"Observed distinct values: {len(proposal.observed_values)}. "
            f"Activation blockers: {len(proposal.activation_blockers)}."
        ),
        required_declarations=list(_REQUIRED_CONTRACT_FIELDS),
        activation_checklist=checklist,
        forbidden_shortcuts=[
            "Do not infer labels from file names, topology, or artifacts.",
            "Do not fabricate targets outside declared positive/negative values.",
            "sedated does not imply no_experience.",
            "unresponsive does not imply unconscious.",
            "Do not claim consciousness, self, soul, liberation, afterlife, or ontology proof.",
            "Do not activate real contract without explicit human-reviewed PR.",
        ],
        reviewer_instructions=(
            "Review activation_proposal.json. "
            "Declare explicit_label_column, positive_values, negative_values, "
            "label_scope, and join_keys in a new contract activation PR. "
            "Do NOT merge that PR without peer review. "
            "Ensure all activation_checklist items are true before activating."
        ),
    )


def scan_for_banned_phrases(text: str) -> list[str]:
    lower = text.lower()
    return [p for p in _BANNED_PHRASES if p in lower]


def write_activation_outputs(
    proposal: ActivationProposal,
    audit_rows: list[MetadataValueAuditRow],
    out_dir: str,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    packet = build_human_review_packet(proposal)

    blockers_data = {
        "dataset_id": proposal.dataset_id,
        "n_blockers": len(proposal.activation_blockers),
        "blockers": proposal.activation_blockers,
        "contract_activation_allowed": False,
    }

    omega = {
        "event_type": "ds005620_contract_activation_audit",
        "dataset_id": proposal.dataset_id,
        "safe_claim": _SAFE_CLAIM,
        "contract_activation_allowed": False,
        "human_review_required": True,
    }

    report_lines = [
        "# DS005620 Contract Activation Audit (P16)",
        "",
        "## Safe Claim",
        "",
        _SAFE_CLAIM,
        "",
        "## Purpose",
        "",
        "This report documents a read-only audit of local DS005620 metadata files for",
        "explicit human-reviewed label-contract activation readiness.",
        "No labels were inferred. No targets were fabricated. No data was downloaded.",
        "",
        "## Activation Gates",
        "",
    ]
    for gate_name, gate_val in proposal.gates.to_dict().items():
        status = "PASS" if gate_val else "BLOCKED"
        report_lines.append(f"- `{gate_name}`: {status}")
    report_lines += [
        "",
        "## Activation Blockers",
        "",
    ]
    for b in proposal.activation_blockers:
        report_lines.append(f"- {b}")
    report_lines += [
        "",
        "## Candidate Metadata",
        "",
        f"- File: `{proposal.candidate_metadata_file or 'none found'}`",
        f"- Label column: `{proposal.candidate_label_column or 'none detected'}`",
        f"- Observed values: {proposal.observed_values}",
        f"- Ambiguous values: {proposal.ambiguous_values_found}",
        "",
        "## How to Activate a Real Contract",
        "",
        "Real contract activation requires a separate human-reviewed PR with:",
        "",
        "1. `explicit_label_column` declared",
        "2. `positive_values` declared",
        "3. `negative_values` declared",
        "4. `label_scope` declared (`window`, `file`, `subject`, or `session`)",
        "5. `join_keys` confirmed",
        "6. Peer review completed before merge",
        "",
        "## Guardrails",
        "",
        "- No label inference from file names, topology, or artifacts",
        "- No target fabrication",
        "- sedated does not imply no_experience",
        "- unresponsive does not imply unconscious",
        "- No consciousness, self, soul, liberation, afterlife, or ontology proof claims",
        "- contract_activation_allowed is always false in this audit",
    ]
    report_text = "\n".join(report_lines) + "\n"

    hits = scan_for_banned_phrases(report_text)
    if hits:
        raise ValueError(f"Banned phrases found in report: {hits}")

    paths: dict[str, str] = {}

    p = out / "activation_proposal.json"
    p.write_text(json.dumps(proposal.to_dict(), indent=2), encoding="utf-8")
    paths["activation_proposal"] = str(p)

    p = out / "human_review_packet.json"
    p.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")
    paths["human_review_packet"] = str(p)

    p = out / "activation_blockers.json"
    p.write_text(json.dumps(blockers_data, indent=2), encoding="utf-8")
    paths["activation_blockers"] = str(p)

    p = out / "omega_event.json"
    p.write_text(json.dumps(omega, indent=2), encoding="utf-8")
    paths["omega_event"] = str(p)

    p = out / "report.md"
    p.write_text(report_text, encoding="utf-8")
    paths["report"] = str(p)

    p = out / "metadata_value_audit.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["column", "value", "count", "is_ambiguous",
                         "candidate_positive", "candidate_negative"],
        )
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(asdict(row))
    paths["metadata_value_audit"] = str(p)

    return paths
