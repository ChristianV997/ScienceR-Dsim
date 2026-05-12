"""Tests for P17.1 — DS005620 reviewed contract activation materializer.

32 tests covering:
  - load functions (3)
  - materialize valid declaration (6)
  - materialize invalid declaration (5)
  - p12_external_contract shape (4)
  - p18_handoff (4)
  - write outputs (4)
  - omega_event (3)
  - module isolation / guardrails (3)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from sciencer_d.btc_icft.labels.ds005620_activation_declaration import _REQUIRED_JOIN_KEYS
from sciencer_d.btc_icft.labels.ds005620_reviewed_contract_activation import (
    _BANNED_PHRASES,
    _FORBIDDEN_CLAIMS,
    _SAFE_CLAIM,
    DS005620ReviewedContractActivationResult,
    build_p12_external_contract,
    build_p18_handoff,
    build_reviewed_contract_omega_event,
    load_declaration_validation,
    load_validated_declaration,
    materialize_reviewed_contract,
    write_reviewed_contract_outputs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_DECLARATION = {
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
        "in the study design."
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
    "notes": ["Test fixture."],
}

_INVALID_DECLARATION = {
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


# ---------------------------------------------------------------------------
# Section 1: Load functions (3 tests)
# ---------------------------------------------------------------------------

def test_load_validated_declaration_missing_file():
    with pytest.raises(FileNotFoundError):
        load_validated_declaration("/nonexistent/path/declaration.json")


def test_load_declaration_validation_missing_file():
    with pytest.raises(FileNotFoundError):
        load_declaration_validation("/nonexistent/path/validation.json")


def test_load_validated_declaration_roundtrip(tmp_path):
    p = tmp_path / "declaration.json"
    p.write_text(json.dumps(_VALID_DECLARATION), encoding="utf-8")
    loaded = load_validated_declaration(str(p))
    assert loaded["dataset_id"] == "DS005620"
    assert loaded["explicit_label_column"] == "trial_type"


# ---------------------------------------------------------------------------
# Section 2: materialize_reviewed_contract — valid declaration (6 tests)
# ---------------------------------------------------------------------------

def test_materialize_valid_reviewed_contract_valid():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.reviewed_contract_valid is True


def test_materialize_valid_activation_allowed():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.activation_allowed is True


def test_materialize_valid_no_errors():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.activation_errors == []


def test_materialize_valid_contract_status():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.p12_external_contract["contract_status"] == "active_reviewed_external_contract"


def test_materialize_valid_safe_claim():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.safe_claim == _SAFE_CLAIM
    assert "labels or targets" in result.safe_claim


def test_materialize_valid_forbidden_claims():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert isinstance(result.forbidden_claims, list)
    assert len(result.forbidden_claims) > 0


# ---------------------------------------------------------------------------
# Section 3: materialize_reviewed_contract — invalid declaration (5 tests)
# ---------------------------------------------------------------------------

def test_materialize_invalid_reviewed_contract_valid():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    assert result.reviewed_contract_valid is False


def test_materialize_invalid_activation_allowed():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    assert result.activation_allowed is False


def test_materialize_invalid_has_errors():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    assert len(result.activation_errors) > 0


def test_materialize_invalid_contract_status():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    assert result.p12_external_contract["contract_status"] == "blocked_invalid_declaration"


def test_materialize_invalid_p18_not_ready():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    p18 = result.p18_handoff
    assert p18["ready_for_p12_alignment"] is False
    assert p18["ready_for_p13_target_injection"] is False
    assert p18["ready_for_p11_target_aware_benchmark"] is False


# ---------------------------------------------------------------------------
# Section 4: p12_external_contract shape (4 tests)
# ---------------------------------------------------------------------------

def test_p12_contract_required_fields():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    prc = result.p12_external_contract
    for field in (
        "dataset_id", "contract_status", "explicit_label_column",
        "positive_values", "negative_values", "label_scope",
        "join_keys", "metadata_provenance", "activation_provenance", "guardrails",
    ):
        assert field in prc, f"Missing P12-compatible field: {field}"


def test_p12_contract_label_column():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.p12_external_contract["explicit_label_column"] == "trial_type"


def test_p12_contract_values():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    prc = result.p12_external_contract
    assert "focus" in prc["positive_values"]
    assert "mind_wandering" in prc["negative_values"]


def test_p12_contract_guardrails():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert isinstance(result.p12_external_contract["guardrails"], list)
    assert len(result.p12_external_contract["guardrails"]) > 0


# ---------------------------------------------------------------------------
# Section 5: p18_handoff (4 tests)
# ---------------------------------------------------------------------------

def test_p18_handoff_ready_when_valid():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    p18 = result.p18_handoff
    assert p18["ready_for_p12_alignment"] is True
    assert p18["ready_for_p13_target_injection"] is True
    assert p18["ready_for_p11_target_aware_benchmark"] is True


def test_p18_handoff_not_ready_when_invalid():
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    p18 = result.p18_handoff
    assert p18["ready_for_p12_alignment"] is False
    assert p18["ready_for_p13_target_injection"] is False
    assert p18["ready_for_p11_target_aware_benchmark"] is False


def test_p18_handoff_required_inputs():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    inputs = result.p18_handoff.get("required_inputs_for_p18", [])
    assert isinstance(inputs, list)
    assert len(inputs) > 0


def test_p18_handoff_no_blockers_when_valid():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.p18_handoff.get("blockers", []) == []


# ---------------------------------------------------------------------------
# Section 6: write outputs (4 tests)
# ---------------------------------------------------------------------------

def test_write_outputs_writes_6_files(tmp_path):
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    paths = write_reviewed_contract_outputs(result, str(tmp_path))
    assert len(paths) == 6
    for key, path in paths.items():
        assert Path(path).is_file(), f"Expected file: {key} → {path}"


def test_write_outputs_reviewed_contract_status(tmp_path):
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    write_reviewed_contract_outputs(result, str(tmp_path))
    data = json.loads((tmp_path / "reviewed_contract.json").read_text())
    assert data.get("status") == "active_reviewed_external_contract" or \
           data.get("contract_status") == "active_reviewed_external_contract"
    assert data.get("activation_allowed") is True


def test_write_outputs_reviewed_activation_report_guardrails(tmp_path):
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    write_reviewed_contract_outputs(result, str(tmp_path))
    data = json.loads((tmp_path / "reviewed_activation_report.json").read_text())
    assert data.get("no_source_contract_modified") is True
    assert data.get("no_p11_run") is True
    assert data.get("no_label_inference") is True
    assert data.get("no_target_fabrication") is True


def test_write_outputs_invalid_writes_blocked_status(tmp_path):
    result = materialize_reviewed_contract(_INVALID_DECLARATION)
    write_reviewed_contract_outputs(result, str(tmp_path))
    data = json.loads((tmp_path / "reviewed_contract.json").read_text())
    assert data.get("status") == "blocked_invalid_declaration" or \
           data.get("activation_allowed") is False


# ---------------------------------------------------------------------------
# Section 7: omega_event (3 tests)
# ---------------------------------------------------------------------------

def test_omega_event_has_event_id():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    omega = result.omega_event
    assert "event_id" in omega
    assert len(omega["event_id"]) == 16


def test_omega_event_source_contracts_not_modified():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.omega_event.get("source_contracts_modified") is False


def test_omega_event_no_p11_run():
    result = materialize_reviewed_contract(_VALID_DECLARATION)
    assert result.omega_event.get("p11_run") is False


# ---------------------------------------------------------------------------
# Section 8: module isolation / guardrails (3 tests)
# ---------------------------------------------------------------------------

def test_safe_claim_no_banned_phrases():
    lower = _SAFE_CLAIM.lower()
    for phrase in _BANNED_PHRASES:
        assert phrase not in lower, f"Banned phrase found in _SAFE_CLAIM: {phrase!r}"


def test_forbidden_claims_non_empty():
    assert isinstance(_FORBIDDEN_CLAIMS, list)
    assert len(_FORBIDDEN_CLAIMS) > 0
    assert any("consciousness" in c.lower() for c in _FORBIDDEN_CLAIMS)


def test_materialize_with_validation_cross_check():
    mock_validation = {
        "dataset_id": "DS005620",
        "declaration_valid": True,
        "activation_dry_run_allowed": True,
        "real_contract_activation_allowed": False,
        "validation_errors": [],
        "validation_warnings": ["Some cross-check warning."],
        "normalized_declaration": _VALID_DECLARATION,
    }
    result = materialize_reviewed_contract(_VALID_DECLARATION, mock_validation)
    assert result.reviewed_contract_valid is True
    assert result.activation_allowed is True


# ---------------------------------------------------------------------------
# CLI integration tests (via subprocess-free argv injection)
# ---------------------------------------------------------------------------

def test_cli_no_args_returns_1():
    from sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract import main
    rc = main(["--out", "/tmp/p171_no_arg_test"])
    assert rc == 1


def test_cli_mock_valid_returns_0(tmp_path):
    from sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract import main
    rc = main(["--mock-valid-declaration", "--out", str(tmp_path)])
    assert rc == 0


def test_cli_mock_invalid_returns_0(tmp_path):
    from sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract import main
    rc = main(["--mock-invalid-declaration", "--out", str(tmp_path)])
    assert rc == 0


def test_cli_mock_valid_writes_6_files(tmp_path):
    from sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract import main
    main(["--mock-valid-declaration", "--out", str(tmp_path)])
    files = list(tmp_path.iterdir())
    assert len(files) == 6


def test_cli_mock_invalid_writes_blocked_contract(tmp_path):
    from sciencer_d.btc_icft.pipelines.materialize_ds005620_reviewed_contract import main
    main(["--mock-invalid-declaration", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "reviewed_contract.json").read_text())
    assert data.get("activation_allowed") is False or \
           data.get("status") == "blocked_invalid_declaration"
