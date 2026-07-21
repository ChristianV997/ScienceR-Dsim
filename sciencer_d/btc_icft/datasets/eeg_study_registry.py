"""EEG study dataset registry and readiness inspection utilities.

This module supports local dataset readiness checks only.
It does not download data or extract signal features.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sciencer_d.btc_icft.io import build_reader_capability_report, inspect_eeg_files


@dataclass
class StudyDatasetRecord:
    dataset_id: str
    title: str
    source_hint: str
    expected_modality: str
    expected_formats: list[str]
    local_root_hints: list[str]
    label_contract_caveats: list[str]
    intended_use: str
    guardrails: list[str]


@dataclass
class StudyDatasetReadiness:
    dataset_id: str
    source_hint: str
    local_root: str
    files_inspected: list[str] = field(default_factory=list)
    readable_files: list[str] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    reader_adapters_used: list[str] = field(default_factory=list)
    readiness_status: str = "missing_root"
    next_required_step: str = "Provide a valid local dataset root"
    notes: list[str] = field(default_factory=list)


def get_study_registry() -> dict[str, StudyDatasetRecord]:
    records = [
        StudyDatasetRecord("DS005620", "OpenNeuro meditation EEG study DS005620", "OpenNeuro", "EEG", [".edf", ".set", ".vhdr", ".csv", ".tsv"], ["./data/ds005620", "./inputs/ds005620"], ["No unsafe label inference from filenames.", "Respect study-specific event/condition contracts."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
        StudyDatasetRecord("DS002094", "OpenNeuro EEG study DS002094", "OpenNeuro", "EEG", [".edf", ".set", ".vhdr", ".csv", ".tsv"], ["./data/ds002094", "./inputs/ds002094"], ["No unsafe label inference from participant identifiers."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
        StudyDatasetRecord("ds001787", "OpenNeuro meditation EEG dataset ds001787", "OpenNeuro", "EEG", [".edf", ".set", ".vhdr", ".csv", ".tsv"], ["./data/ds001787", "./inputs/ds001787"], ["No unsafe label inference from meditation session names."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
        StudyDatasetRecord("ds003969", "OpenNeuro meditation EEG dataset ds003969", "OpenNeuro", "EEG", [".edf", ".set", ".vhdr", ".csv", ".tsv"], ["./data/ds003969", "./inputs/ds003969"], ["No unsafe label inference across cohorts."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
        StudyDatasetRecord("ds003816", "OpenNeuro meditation EEG dataset ds003816", "OpenNeuro", "EEG", [".edf", ".set", ".vhdr", ".csv", ".tsv"], ["./data/ds003816", "./inputs/ds003816"], ["No unsafe label inference from condition names."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
        StudyDatasetRecord("PhysioNet_GABA", "PhysioNet GABA EEG study", "PhysioNet", "EEG", [".edf", ".bdf", ".fif", ".csv", ".tsv"], ["./data/physionet_gaba", "./inputs/physionet_gaba"], ["No unsafe label inference for intervention/control mapping."], "Reader compatibility and dataset readiness for future signal-level feature extraction.", ["no_data_download", "no_model_training", "no_level_m_feature_extraction", "no_level_t_topology", "no_level_o_c_q", "no_ontology_proof_claims"]),
    ]
    return {r.dataset_id: r for r in records}


def get_study_record(dataset_id: str) -> StudyDatasetRecord:
    registry = get_study_registry()
    if dataset_id not in registry:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    return registry[dataset_id]


def discover_study_files(local_root: str) -> list[str]:
    root = Path(local_root)
    if not root.exists() or not root.is_dir():
        return []
    files = [str(p) for p in root.rglob("*") if p.is_file()]
    return sorted(files)


def inspect_study_dataset(dataset_id: str, local_root: str) -> StudyDatasetReadiness:
    record = get_study_record(dataset_id)
    root = Path(local_root)
    if not root.exists() or not root.is_dir():
        return StudyDatasetReadiness(dataset_id=record.dataset_id, source_hint=record.source_hint, local_root=str(root), readiness_status="missing_root", next_required_step="Provide an existing local dataset root or use --mock-fixture.")

    files = discover_study_files(str(root))
    if not files:
        return StudyDatasetReadiness(dataset_id=record.dataset_id, source_hint=record.source_hint, local_root=str(root), readiness_status="no_candidate_files", next_required_step="Place study files under the local root and rerun inspection.")

    rows = inspect_eeg_files(files)
    readable_files = [r.path for r in rows if r.readable]
    unsupported = [r.path for r in rows if r.status in {"unsupported_or_dependency_missing", "unknown_extension"}]
    adapters = sorted({r.adapter for r in rows if r.adapter})

    if readable_files and all(Path(f).suffix.lower() in {".csv", ".txt", ".tsv"} for f in readable_files):
        status = "fixture_readable"
        next_step = "Provide real study EEG formats or proceed to staged P9 signal extraction planning."
    elif readable_files:
        status = "ready_for_p9_signal_extraction"
        next_step = "Proceed with controlled P9 signal-level feature extraction implementation."
    elif unsupported:
        status = "unsupported_or_dependency_missing"
        next_step = "Install optional reader dependency (e.g., mne) or convert to supported fixture formats."
    else:
        status = "partially_readable"
        next_step = "Review file readability errors and repair malformed files."

    return StudyDatasetReadiness(dataset_id=record.dataset_id, source_hint=record.source_hint, local_root=str(root), files_inspected=files, readable_files=readable_files, unsupported_files=unsupported, reader_adapters_used=adapters, readiness_status=status, next_required_step=next_step)


def write_study_dataset_outputs(result: StudyDatasetReadiness, out_dir: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = get_study_record(result.dataset_id)
    outputs: dict[str, str] = {}

    study_card = {"study": asdict(record), "safe_claim": "The local study dataset was inspected for reader compatibility and readiness for future signal-level feature extraction."}
    p = out / "study_card.json"
    p.write_text(json.dumps(study_card, indent=2))
    outputs["study_card"] = str(p)

    readability = {"dataset_id": result.dataset_id, "local_root": result.local_root, "files_inspected": result.files_inspected, "readable_files": result.readable_files, "unsupported_files": result.unsupported_files}
    p = out / "file_readability_report.json"
    p.write_text(json.dumps(readability, indent=2))
    outputs["file_readability_report"] = str(p)

    caps = build_reader_capability_report()
    p = out / "reader_capability_report.json"
    p.write_text(json.dumps(caps, indent=2))
    outputs["reader_capability_report"] = str(p)

    p = out / "dataset_readiness_report.json"
    p.write_text(json.dumps(asdict(result), indent=2))
    outputs["dataset_readiness_report"] = str(p)

    md = f"# EEG Study Dataset Readiness Report\n\n- dataset ID: {result.dataset_id}\n- source hint: {result.source_hint}\n- local root: {result.local_root}\n- files inspected: {len(result.files_inspected)}\n- readable files: {len(result.readable_files)}\n- unsupported files: {len(result.unsupported_files)}\n- reader adapters used: {result.reader_adapters_used}\n- readiness status: {result.readiness_status}\n- next required step: {result.next_required_step}\n\nThe local study dataset was inspected for reader compatibility and readiness for future signal-level feature extraction.\n"
    p = out / "report.md"
    p.write_text(md)
    outputs["report"] = str(p)

    return outputs
