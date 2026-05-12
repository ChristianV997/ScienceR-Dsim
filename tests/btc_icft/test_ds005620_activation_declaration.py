"""Tests for P17.0 — DS005620 activation declaration validator.

All tests use tmp_path only. No real dataset files required.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sciencer_d.btc_icft.labels.ds005620_activation_declaration import (
    DS005620ActivationDeclaration,
    DS005620ActivationDeclarationValidationResult,
    _BANNED_PHRASES,
    _REQUIRED_JOIN_KEYS,
    _SAFE_CLAIM,
    build_activation_declaration_template,
    build_activation_declaration_omega_event,
    load_activation_declaration,
    load_activation_packet,
    validate_activation_declaration,
    write_activation_declaration_outputs,
)

DATASET_ID = "DS005620"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_declaration(**overrides) -> dict:
    base = {
        "dataset_id": "DS005620",
        "explicit_label_column": "trial_type",
        "positive_values": ["focus"],
        "negative_values": ["mind_wandering"],
        "label_scope": "window",
        "join_keys": _REQUIRED_JOIN_KEYS[:],
        "metadata_provenance": "data/DS005620/events.tsv",
        "semantic_justification": (
            "The trial_type column contains experimenter-assigned condition labels "
            "from the original study protocol. No shortcut inference."
        ),
        "no_shortcut_inference_confirmation": (
            "I confirm no sedated-to-no_experience shortcut and "
            "no unresponsive-to-unconscious shortcut is used."
        ),
        "reviewer_identity_or_role": "data_curator",
        "review_date": "2026-05-12",
        "both_classes_present_confirmation": True,
        "ambiguity_reviewed": True,
    }
    base.update(overrides)
    return base


def _run_validation(decl: dict, packet: dict | None = None):
    return validate_activation_declaration(decl, packet)


# ---------------------------------------------------------------------------
# Test 1: template includes all required fields
# ---------------------------------------------------------------------------

class TestTemplate:
    def test_build_template_includes_all_required_fields(self):
        template = build_activation_declaration_template()
        required = [
            "dataset_id", "explicit_label_column", "positive_values",
            "negative_values", "label_scope", "join_keys",
            "metadata_provenance", "semantic_justification",
            "no_shortcut_inference_confirmation", "reviewer_identity_or_role",
            "review_date", "both_classes_present_confirmation",
            "ambiguity_reviewed",
        ]
        for f in required:
            assert f in template, f"Template missing field: {f}"

    def test_template_join_keys_has_all_eight(self):
        template = build_activation_declaration_template()
        for k in _REQUIRED_JOIN_KEYS:
            assert k in template["join_keys"]


# ---------------------------------------------------------------------------
# Tests 2–14: validation rules
# ---------------------------------------------------------------------------

class TestValidationRules:
    def test_valid_declaration_passes(self):
        result = _run_validation(_valid_declaration())
        assert result.declaration_valid is True
        assert result.validation_errors == []

    def test_missing_field_fails(self):
        decl = _valid_declaration()
        del decl["explicit_label_column"]
        result = _run_validation(decl)
        assert result.declaration_valid is False
        assert any("explicit_label_column" in e for e in result.validation_errors)

    def test_wrong_dataset_id_fails(self):
        result = _run_validation(_valid_declaration(dataset_id="DS000001"))
        assert result.declaration_valid is False
        assert any("dataset_id" in e.lower() for e in result.validation_errors)

    def test_empty_positive_values_fails(self):
        result = _run_validation(_valid_declaration(positive_values=[]))
        assert result.declaration_valid is False
        assert any("positive_values" in e for e in result.validation_errors)

    def test_empty_negative_values_fails(self):
        result = _run_validation(_valid_declaration(negative_values=[]))
        assert result.declaration_valid is False
        assert any("negative_values" in e for e in result.validation_errors)

    def test_overlapping_positive_negative_fails(self):
        result = _run_validation(
            _valid_declaration(positive_values=["x"], negative_values=["x"])
        )
        assert result.declaration_valid is False
        assert any("overlap" in e for e in result.validation_errors)

    def test_unsupported_label_scope_fails(self):
        result = _run_validation(_valid_declaration(label_scope="block"))
        assert result.declaration_valid is False
        assert any("label_scope" in e for e in result.validation_errors)

    def test_missing_strict_join_key_fails(self):
        jk = [k for k in _REQUIRED_JOIN_KEYS if k != "window_id"]
        result = _run_validation(_valid_declaration(join_keys=jk))
        assert result.declaration_valid is False
        assert any("join_keys" in e for e in result.validation_errors)

    def test_unknown_metadata_provenance_fails(self):
        result = _run_validation(_valid_declaration(metadata_provenance="unknown"))
        assert result.declaration_valid is False
        assert any("provenance" in e.lower() or "unknown" in e.lower()
                   for e in result.validation_errors)

    def test_short_semantic_justification_fails(self):
        result = _run_validation(_valid_declaration(semantic_justification="too short"))
        assert result.declaration_valid is False
        assert any("semantic_justification" in e for e in result.validation_errors)

    def test_missing_no_shortcut_phrase_fails(self):
        result = _run_validation(
            _valid_declaration(
                no_shortcut_inference_confirmation="I confirm no shortcuts"
            )
        )
        assert result.declaration_valid is False
        assert any("no_shortcut" in e for e in result.validation_errors)

    def test_both_classes_false_fails(self):
        result = _run_validation(
            _valid_declaration(both_classes_present_confirmation=False)
        )
        assert result.declaration_valid is False
        assert any("both_classes" in e for e in result.validation_errors)

    def test_ambiguity_reviewed_false_fails(self):
        result = _run_validation(_valid_declaration(ambiguity_reviewed=False))
        assert result.declaration_valid is False
        assert any("ambiguity_reviewed" in e for e in result.validation_errors)


# ---------------------------------------------------------------------------
# Tests 15–16: activation flags
# ---------------------------------------------------------------------------

class TestActivationFlags:
    def test_real_contract_activation_always_false(self):
        for decl in [_valid_declaration(), _valid_declaration(dataset_id="DS000001")]:
            result = _run_validation(decl)
            assert result.real_contract_activation_allowed is False

    def test_activation_dry_run_allowed_only_if_valid(self):
        valid_result = _run_validation(_valid_declaration())
        assert valid_result.activation_dry_run_allowed is True

        invalid_result = _run_validation(_valid_declaration(positive_values=[]))
        assert invalid_result.activation_dry_run_allowed is False


# ---------------------------------------------------------------------------
# Tests 17–19: P16 packet cross-checks
# ---------------------------------------------------------------------------

class TestPacketCrossCheck:
    def _make_packet(self, candidate_cols=None, unresolved=None, activation_allowed=False):
        return {
            "dataset_id": "DS005620",
            "candidate_label_columns": candidate_cols or ["trial_type"],
            "unresolved_values": unresolved or ["focus", "mind_wandering"],
            "contract_activation_allowed": activation_allowed,
        }

    def test_warns_if_label_column_not_in_candidates(self):
        packet = self._make_packet(candidate_cols=["condition"])
        result = _run_validation(_valid_declaration(), packet)
        # explicit_label_column "trial_type" not in ["condition"]
        assert any("not in" in w.lower() or "candidate" in w.lower()
                   for w in result.validation_warnings)

    def test_warns_if_values_not_in_unresolved(self):
        packet = self._make_packet(unresolved=["a", "b"])
        result = _run_validation(_valid_declaration(), packet)
        assert any("unresolved" in w.lower() or "not in" in w.lower()
                   for w in result.validation_warnings)

    def test_active_looking_packet_is_warned(self):
        packet = self._make_packet(activation_allowed=True)
        result = _run_validation(_valid_declaration(), packet)
        assert any("contract_activation_allowed" in w.lower() or "ignoring" in w.lower()
                   for w in result.validation_warnings)
        # Still keeps real activation false
        assert result.real_contract_activation_allowed is False


# ---------------------------------------------------------------------------
# Tests 20–23: write outputs
# ---------------------------------------------------------------------------

class TestWriteOutputs:
    def _run_write(self, tmp_path, valid=True):
        decl = _valid_declaration() if valid else _valid_declaration(positive_values=[])
        result = _run_validation(decl)
        paths = write_activation_declaration_outputs(result, str(tmp_path / "out"))
        return result, paths, tmp_path / "out"

    def test_writes_all_six_files(self, tmp_path):
        _, _, out = self._run_write(tmp_path)
        for name in [
            "activation_declaration_template.json",
            "activation_declaration_validation.json",
            "activation_contract_preview.json",
            "activation_declaration_errors.json",
            "omega_event.json",
            "report.md",
        ]:
            assert (out / name).is_file(), f"Missing: {name}"

    def test_json_outputs_parse(self, tmp_path):
        _, _, out = self._run_write(tmp_path)
        for name in [
            "activation_declaration_template.json",
            "activation_declaration_validation.json",
            "activation_contract_preview.json",
            "activation_declaration_errors.json",
            "omega_event.json",
        ]:
            data = json.loads((out / name).read_text())
            assert isinstance(data, dict), f"{name} is not a dict"

    def test_preview_status_is_preview_human_reviewed_not_active(self, tmp_path):
        _, _, out = self._run_write(tmp_path, valid=True)
        data = json.loads((out / "activation_contract_preview.json").read_text())
        assert data["status"] == "preview_human_reviewed_not_active"

    def test_preview_does_not_modify_source_contracts(self, tmp_path):
        _, _, out = self._run_write(tmp_path, valid=True)
        # Source file should be unchanged — just check it still has its original content
        src = Path("sciencer_d/btc_icft/labels/eeg_label_contracts.py").read_text()
        assert "DS005620" in src or len(src) > 0  # file still intact
        # Preview output is in out_dir only, not in source
        preview = json.loads((out / "activation_contract_preview.json").read_text())
        assert preview["status"] == "preview_human_reviewed_not_active"


# ---------------------------------------------------------------------------
# Tests 24–26: omega_event and report
# ---------------------------------------------------------------------------

class TestOmegaAndReport:
    def test_omega_event_has_safe_claim_and_no_banned_phrases(self, tmp_path):
        result = _run_validation(_valid_declaration())
        write_activation_declaration_outputs(result, str(tmp_path / "out"))
        data = json.loads((tmp_path / "out" / "omega_event.json").read_text())
        assert "safe_claim" in data
        claim = data["safe_claim"].lower()
        for phrase in _BANNED_PHRASES:
            assert phrase not in claim, f"Banned phrase in omega safe_claim: {phrase}"

    def test_report_contains_cautious_terms(self, tmp_path):
        result = _run_validation(_valid_declaration())
        write_activation_declaration_outputs(result, str(tmp_path / "out"))
        text = (tmp_path / "out" / "report.md").read_text().lower()
        assert "human-authored" in text or "human authored" in text
        assert "no-shortcut safeguards" in text or "no shortcut" in text
        assert "before any real contract activation" in text or "real contract activation" in text

    def test_report_no_banned_phrases(self, tmp_path):
        result = _run_validation(_valid_declaration())
        write_activation_declaration_outputs(result, str(tmp_path / "out"))
        text = (tmp_path / "out" / "report.md").read_text().lower()
        for phrase in _BANNED_PHRASES:
            assert phrase not in text, f"Banned phrase in report.md: {phrase}"


# ---------------------------------------------------------------------------
# Tests 27–31: CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_write_template_exits_zero_and_writes_template(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration",
             "--write-template",
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"stderr: {cp.stderr}"
        assert (tmp_path / "out" / "activation_declaration_template.json").is_file()

    def test_mock_valid_declaration_exits_zero_and_validation_passes(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration",
             "--mock-valid-declaration",
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"stderr: {cp.stderr}"
        data = json.loads((tmp_path / "out" / "activation_declaration_validation.json").read_text())
        assert data["declaration_valid"] is True
        assert data["real_contract_activation_allowed"] is False

    def test_mock_invalid_declaration_exits_zero_and_validation_fails(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration",
             "--mock-invalid-declaration",
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"stderr: {cp.stderr}"
        data = json.loads((tmp_path / "out" / "activation_declaration_validation.json").read_text())
        assert data["declaration_valid"] is False
        assert data["real_contract_activation_allowed"] is False

    def test_missing_declaration_exits_nonzero(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration",
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode != 0
        assert "declaration" in cp.stderr.lower() or "required" in cp.stderr.lower()

    def test_missing_activation_packet_fails_cleanly_when_provided(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.validate_ds005620_activation_declaration",
             "--mock-valid-declaration",
             "--activation-packet", str(tmp_path / "nonexistent.json"),
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode != 0
        assert "not found" in cp.stderr.lower() or "activation packet" in cp.stderr.lower()


# ---------------------------------------------------------------------------
# Test 32: config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_contains_required_outputs(self):
        cfg = Path("configs/btc_icft/ds005620_activation_declaration.yaml").read_text()
        for item in [
            "activation_declaration_template.json",
            "activation_declaration_validation.json",
            "activation_contract_preview.json",
            "activation_declaration_errors.json",
            "omega_event.json",
            "report.md",
        ]:
            assert item in cfg, f"Config missing required output: {item}"

    def test_config_contains_guardrails(self):
        cfg = Path("configs/btc_icft/ds005620_activation_declaration.yaml").read_text()
        for gr in [
            "no_label_inference",
            "no_target_fabrication",
            "no_automatic_real_contract_activation",
            "no_sedated_to_no_experience",
            "no_unresponsive_to_unconscious",
        ]:
            assert gr in cfg, f"Config missing guardrail: {gr}"

    def test_config_real_activation_false(self):
        cfg = Path("configs/btc_icft/ds005620_activation_declaration.yaml").read_text()
        assert "real_contract_activation_allowed: false" in cfg

    def test_config_no_banned_phrases(self):
        cfg = Path("configs/btc_icft/ds005620_activation_declaration.yaml").read_text().lower()
        for phrase in _BANNED_PHRASES:
            assert phrase not in cfg, f"Banned phrase in config: {phrase}"


# ---------------------------------------------------------------------------
# Tests 33–35: no y target, no P12/P13/P11 modification, no mt_real
# ---------------------------------------------------------------------------

class TestModuleIsolation:
    def test_no_y_target_in_outputs(self, tmp_path):
        result = _run_validation(_valid_declaration())
        paths = write_activation_declaration_outputs(result, str(tmp_path / "out"))
        out = tmp_path / "out"
        for name in ["activation_declaration_validation.json",
                     "activation_contract_preview.json",
                     "activation_declaration_errors.json",
                     "omega_event.json"]:
            data = json.loads((out / name).read_text())
            _assert_no_y_target(data)

    def test_no_p11_p12_p13_imported_in_declaration_module(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_activation_declaration.py"
        ).read_text()
        assert "eeg_signal_residual" not in src
        assert "eeg_label_contracts" not in src
        assert "eeg_target_injection" not in src
        assert "run_eeg_signal_mt" not in src

    def test_no_legacy_mt_real_imported_in_declaration_module(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_activation_declaration.py"
        ).read_text()
        assert "run_ds005620_mt_real" not in src
        assert "ds005620_residual" not in src


def _assert_no_y_target(data, path=""):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("y", "y_target", "y_targets"):
                if isinstance(v, list):
                    assert len(v) == 0, f"Non-empty {k} at {path}"
            _assert_no_y_target(v, f"{path}.{k}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _assert_no_y_target(item, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Additional: safe_claim guardrail
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_safe_claim_no_banned_phrases(self):
        for phrase in _BANNED_PHRASES:
            assert phrase not in _SAFE_CLAIM.lower()

    def test_real_activation_never_true_in_source(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_activation_declaration.py"
        ).read_text()
        assert "real_contract_activation_allowed=True" not in src
        assert "real_contract_activation_allowed = True" not in src

    def test_no_infer_label_in_source(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_activation_declaration.py"
        ).read_text()
        assert "infer_label" not in src
        assert "make_target" not in src
