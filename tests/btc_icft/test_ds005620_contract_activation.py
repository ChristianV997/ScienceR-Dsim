"""Tests for P16 — DS005620 human-reviewed label contract activation path.

All tests use tmp_path only. No real dataset files required.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from sciencer_d.btc_icft.labels.ds005620_contract_activation import (
    ActivationGates,
    ActivationProposal,
    HumanReviewPacket,
    MetadataValueAuditRow,
    _BANNED_PHRASES,
    _SAFE_CLAIM,
    audit_ds005620_metadata,
    build_human_review_packet,
    scan_for_banned_phrases,
    write_activation_outputs,
)

DATASET_ID = "DS005620"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_events_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["onset", "duration", "trial_type"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_events_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["onset", "duration", "trial_type"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _standard_rows(cond_a="condition_a", cond_b="condition_b") -> list[dict]:
    return [
        {"onset": "0.0", "duration": "1.0", "trial_type": cond_a},
        {"onset": "1.0", "duration": "1.0", "trial_type": cond_b},
        {"onset": "2.0", "duration": "1.0", "trial_type": cond_a},
        {"onset": "3.0", "duration": "1.0", "trial_type": cond_b},
    ]


# ---------------------------------------------------------------------------
# TestActivationGates
# ---------------------------------------------------------------------------

class TestActivationGates:
    def test_default_contract_activation_allowed_is_false(self):
        gates = ActivationGates()
        assert gates.contract_activation_allowed is False

    def test_default_human_review_required_is_true(self):
        gates = ActivationGates()
        assert gates.human_review_required is True

    def test_all_structural_gates_false_by_default(self):
        gates = ActivationGates()
        assert gates.all_structural_gates_passed() is False

    def test_to_dict_has_all_gate_keys(self):
        gates = ActivationGates()
        d = gates.to_dict()
        for key in [
            "metadata_file_exists",
            "explicit_label_column_declared",
            "positive_values_declared",
            "negative_values_declared",
            "label_scope_declared",
            "join_keys_declared",
            "both_classes_present",
            "ambiguous_values_rejected",
            "human_review_required",
            "contract_activation_allowed",
        ]:
            assert key in d, f"Missing gate key: {key}"

    def test_all_structural_gates_passed_when_all_true(self):
        gates = ActivationGates(
            metadata_file_exists=True,
            explicit_label_column_declared=True,
            positive_values_declared=True,
            negative_values_declared=True,
            label_scope_declared=True,
            join_keys_declared=True,
            both_classes_present=True,
            ambiguous_values_rejected=True,
        )
        assert gates.all_structural_gates_passed() is True

    def test_contract_activation_still_false_when_structural_gates_pass(self):
        gates = ActivationGates(
            metadata_file_exists=True,
            explicit_label_column_declared=True,
            positive_values_declared=True,
            negative_values_declared=True,
            label_scope_declared=True,
            join_keys_declared=True,
            both_classes_present=True,
            ambiguous_values_rejected=True,
        )
        assert gates.contract_activation_allowed is False


# ---------------------------------------------------------------------------
# TestAuditDs005620Metadata
# ---------------------------------------------------------------------------

class TestAuditDs005620Metadata:
    def test_returns_proposal_and_audit_rows(self, tmp_path):
        proposal, audit_rows = audit_ds005620_metadata(str(tmp_path))
        assert isinstance(proposal, ActivationProposal)
        assert isinstance(audit_rows, list)

    def test_metadata_file_not_found_when_empty_dir(self, tmp_path):
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.gates.metadata_file_exists is False
        assert proposal.candidate_metadata_file is None

    def test_finds_tsv_events_file(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.gates.metadata_file_exists is True
        assert proposal.candidate_metadata_file is not None

    def test_finds_csv_events_file(self, tmp_path):
        _write_events_csv(tmp_path / "events.csv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.gates.metadata_file_exists is True

    def test_detects_trial_type_column(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.candidate_label_column == "trial_type"

    def test_observed_values_populated(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert "condition_a" in proposal.observed_values
        assert "condition_b" in proposal.observed_values

    def test_audit_rows_populated(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        _, audit_rows = audit_ds005620_metadata(str(tmp_path))
        assert len(audit_rows) > 0
        assert all(isinstance(r, MetadataValueAuditRow) for r in audit_rows)

    def test_contract_activation_always_false(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.gates.contract_activation_allowed is False

    def test_human_review_required_always_true(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.gates.human_review_required is True

    def test_safe_claim_in_proposal(self, tmp_path):
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert proposal.safe_claim == _SAFE_CLAIM

    def test_activation_blockers_present_when_no_metadata(self, tmp_path):
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert len(proposal.activation_blockers) > 0

    def test_activation_blockers_present_even_with_metadata(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert len(proposal.activation_blockers) > 0

    def test_declared_label_column_sets_gate(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(
            str(tmp_path),
            declared_label_column="trial_type",
        )
        assert proposal.gates.explicit_label_column_declared is True

    def test_declared_positive_negative_sets_gates(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(
            str(tmp_path),
            declared_label_column="trial_type",
            declared_positive_values=["condition_a"],
            declared_negative_values=["condition_b"],
        )
        assert proposal.gates.positive_values_declared is True
        assert proposal.gates.negative_values_declared is True

    def test_declared_label_scope_sets_gate(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(
            str(tmp_path),
            declared_label_scope="window",
        )
        assert proposal.gates.label_scope_declared is True

    def test_declared_join_keys_sets_gate(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(
            str(tmp_path),
            declared_join_keys=["dataset_id", "row_id", "window_id"],
        )
        assert proposal.gates.join_keys_declared is True

    def test_both_classes_present_when_pos_neg_found(self, tmp_path):
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(
            str(tmp_path),
            declared_label_column="trial_type",
            declared_positive_values=["condition_a"],
            declared_negative_values=["condition_b"],
        )
        assert proposal.gates.both_classes_present is True

    def test_ambiguous_values_detected(self, tmp_path):
        rows = _standard_rows() + [{"onset": "4.0", "duration": "1.0", "trial_type": "n/a"}]
        _write_events_tsv(tmp_path / "events.tsv", rows)
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        assert "n/a" in proposal.ambiguous_values_found

    def test_no_banned_phrases_in_proposal_notes(self, tmp_path):
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        for note in proposal.notes:
            for phrase in _BANNED_PHRASES:
                assert phrase not in note.lower()


# ---------------------------------------------------------------------------
# TestBuildHumanReviewPacket
# ---------------------------------------------------------------------------

class TestBuildHumanReviewPacket:
    def _make_proposal(self, tmp_path) -> ActivationProposal:
        _write_events_tsv(tmp_path / "events.tsv", _standard_rows())
        proposal, _ = audit_ds005620_metadata(str(tmp_path))
        return proposal

    def test_returns_human_review_packet(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        assert isinstance(packet, HumanReviewPacket)

    def test_safe_claim_in_packet(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        assert packet.safe_claim == _SAFE_CLAIM

    def test_required_declarations_present(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        for field in ["explicit_label_column", "positive_values", "negative_values",
                      "label_scope", "join_keys"]:
            assert field in packet.required_declarations

    def test_activation_checklist_has_items(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        assert len(packet.activation_checklist) >= 8

    def test_forbidden_shortcuts_no_banned_phrases(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        for shortcut in packet.forbidden_shortcuts:
            for phrase in _BANNED_PHRASES:
                assert phrase not in shortcut.lower()

    def test_to_dict_serializable(self, tmp_path):
        proposal = self._make_proposal(tmp_path)
        packet = build_human_review_packet(proposal)
        d = packet.to_dict()
        json.dumps(d)


# ---------------------------------------------------------------------------
# TestScanForBannedPhrases
# ---------------------------------------------------------------------------

class TestScanForBannedPhrases:
    def test_no_hits_in_clean_text(self):
        hits = scan_for_banned_phrases("This is a clean audit report.")
        assert hits == []

    def test_detects_banned_phrase(self):
        hits = scan_for_banned_phrases("This proves consciousness.")
        assert len(hits) > 0

    def test_safe_claim_has_no_banned_phrases(self):
        hits = scan_for_banned_phrases(_SAFE_CLAIM)
        assert hits == []


# ---------------------------------------------------------------------------
# TestWriteActivationOutputs
# ---------------------------------------------------------------------------

class TestWriteActivationOutputs:
    def _run(self, tmp_path):
        _write_events_tsv(tmp_path / "ds" / "events.tsv", _standard_rows())
        proposal, audit_rows = audit_ds005620_metadata(str(tmp_path / "ds"))
        paths = write_activation_outputs(proposal, audit_rows, str(tmp_path / "out"))
        return proposal, audit_rows, paths, tmp_path / "out"

    def test_writes_all_six_outputs(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        for name in [
            "activation_proposal.json",
            "human_review_packet.json",
            "metadata_value_audit.csv",
            "activation_blockers.json",
            "omega_event.json",
            "report.md",
        ]:
            assert (out / name).is_file(), f"Missing: {name}"

    def test_activation_proposal_parses(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "activation_proposal.json").read_text())
        assert "dataset_id" in data
        assert "gates" in data
        assert "activation_blockers" in data

    def test_activation_proposal_contract_activation_always_false(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "activation_proposal.json").read_text())
        assert data["gates"]["contract_activation_allowed"] is False

    def test_human_review_packet_has_safe_claim(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "human_review_packet.json").read_text())
        assert "safe_claim" in data
        assert _SAFE_CLAIM in data["safe_claim"]

    def test_activation_blockers_has_blockers_key(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "activation_blockers.json").read_text())
        assert "blockers" in data
        assert "contract_activation_allowed" in data
        assert data["contract_activation_allowed"] is False

    def test_omega_event_has_safe_claim(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "omega_event.json").read_text())
        assert "safe_claim" in data

    def test_omega_event_contract_activation_false(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        data = json.loads((out / "omega_event.json").read_text())
        assert data["contract_activation_allowed"] is False

    def test_report_md_no_banned_phrases(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        text = (out / "report.md").read_text().lower()
        for phrase in _BANNED_PHRASES:
            assert phrase not in text, f"Banned phrase in report.md: {phrase}"

    def test_report_md_has_safe_claim(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        text = (out / "report.md").read_text()
        assert "DS005620" in text

    def test_metadata_value_audit_csv_has_rows(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        text = (out / "metadata_value_audit.csv").read_text()
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) >= 1

    def test_metadata_value_audit_csv_has_header(self, tmp_path):
        _, _, paths, out = self._run(tmp_path)
        text = (out / "metadata_value_audit.csv").read_text()
        header = text.splitlines()[0]
        assert "column" in header
        assert "value" in header
        assert "is_ambiguous" in header


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_contains_required_outputs(self):
        cfg = Path("configs/btc_icft/ds005620_contract_activation.yaml").read_text()
        for item in [
            "activation_proposal.json",
            "human_review_packet.json",
            "metadata_value_audit.csv",
            "activation_blockers.json",
            "omega_event.json",
            "report.md",
        ]:
            assert item in cfg

    def test_config_contains_activation_gates(self):
        cfg = Path("configs/btc_icft/ds005620_contract_activation.yaml").read_text()
        for gate in [
            "metadata_file_exists",
            "explicit_label_column_declared",
            "positive_values_declared",
            "negative_values_declared",
            "label_scope_declared",
            "join_keys_declared",
        ]:
            assert gate in cfg

    def test_config_contract_activation_allowed_false(self):
        cfg = Path("configs/btc_icft/ds005620_contract_activation.yaml").read_text()
        assert "contract_activation_allowed: false" in cfg

    def test_config_contains_guardrails(self):
        cfg = Path("configs/btc_icft/ds005620_contract_activation.yaml").read_text()
        for gr in [
            "no_label_inference",
            "no_target_fabrication",
            "no_sedated_to_no_experience",
            "no_unresponsive_to_unconscious",
            "no_automatic_real_contract_activation",
        ]:
            assert gr in cfg

    def test_config_no_banned_phrases(self):
        cfg = Path("configs/btc_icft/ds005620_contract_activation.yaml").read_text().lower()
        for phrase in _BANNED_PHRASES:
            assert phrase not in cfg


# ---------------------------------------------------------------------------
# TestCli
# ---------------------------------------------------------------------------

class TestCli:
    def test_help_works(self):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation",
             "--help"],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0
        assert "mock-fixture" in cp.stdout.lower() or "ds-root" in cp.stdout.lower()

    def test_mock_fixture_exits_zero(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation",
             "--mock-fixture",
             "--ds-root", str(tmp_path / "ds"),
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0, f"stderr: {cp.stderr}"

    def test_mock_fixture_writes_all_outputs(self, tmp_path):
        subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation",
             "--mock-fixture",
             "--ds-root", str(tmp_path / "ds"),
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        out = tmp_path / "out"
        for name in [
            "activation_proposal.json",
            "human_review_packet.json",
            "metadata_value_audit.csv",
            "activation_blockers.json",
            "omega_event.json",
            "report.md",
        ]:
            assert (out / name).is_file(), f"Missing: {name}"

    def test_mock_fixture_contract_activation_always_false(self, tmp_path):
        subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation",
             "--mock-fixture",
             "--ds-root", str(tmp_path / "ds"),
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        data = json.loads((tmp_path / "out" / "activation_proposal.json").read_text())
        assert data["gates"]["contract_activation_allowed"] is False

    def test_no_metadata_exits_zero_with_blockers(self, tmp_path):
        cp = subprocess.run(
            [sys.executable, "-m",
             "sciencer_d.btc_icft.pipelines.prepare_ds005620_contract_activation",
             "--ds-root", str(tmp_path / "ds"),
             "--out", str(tmp_path / "out")],
            capture_output=True, text=True,
        )
        assert cp.returncode == 0
        data = json.loads((tmp_path / "out" / "activation_blockers.json").read_text())
        assert data["n_blockers"] > 0


# ---------------------------------------------------------------------------
# TestGuardrailIntegrity
# ---------------------------------------------------------------------------

class TestGuardrailIntegrity:
    def test_safe_claim_has_no_banned_phrases(self):
        for phrase in _BANNED_PHRASES:
            assert phrase not in _SAFE_CLAIM.lower()

    def test_no_label_inference_in_module(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_contract_activation.py"
        ).read_text()
        assert "infer_label" not in src
        assert "make_target" not in src

    def test_contract_activation_always_false_in_source(self):
        src = Path(
            "sciencer_d/btc_icft/labels/ds005620_contract_activation.py"
        ).read_text()
        assert '"contract_activation_allowed": True' not in src
        assert '"contract_activation_allowed": true' not in src

    def test_no_p13_imports_in_pipeline(self):
        src = Path(
            "sciencer_d/btc_icft/pipelines/prepare_ds005620_contract_activation.py"
        ).read_text()
        assert "eeg_target_injection" not in src
        assert "inject_eeg_targets" not in src

    def test_no_p11_imports_in_pipeline(self):
        src = Path(
            "sciencer_d/btc_icft/pipelines/prepare_ds005620_contract_activation.py"
        ).read_text()
        assert "eeg_signal_residual" not in src
        assert "run_eeg_signal_mt" not in src
