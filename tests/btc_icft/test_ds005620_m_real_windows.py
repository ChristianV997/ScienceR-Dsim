from __future__ import annotations

import json
import math
import subprocess
import sys

from sciencer_d.btc_icft.level_m.ds005620_windows import (
    LevelMWindowRow,
    build_level_m_windows_from_bids_inventory,
    build_mock_level_m_windows_from_inspection,
    build_window_artifact_report,
    build_window_leakage_report,
    evaluate_level_m_windows,
    extract_level_m_window_features,
    load_bids_inspection_outputs,
    write_level_m_window_outputs,
)


def _minimal_inspection(tmp_path):
    d = tmp_path / "insp"
    d.mkdir()
    inv = {
        "eeg_candidates": [
            {"path": "a.edf", "relative_path": "sub-01/ses-01/eeg/sub-01_task-awake_run-01_eeg.edf", "subject_id": "sub-01", "session_id": "ses-01", "run_id": "01", "task_label": "awake", "is_eeg_candidate": True},
            {"path": "b.edf", "relative_path": "sub-02/ses-01/eeg/sub-02_task-sedated_run-01_eeg.edf", "subject_id": "sub-02", "session_id": "ses-01", "run_id": "01", "task_label": "sedated", "is_eeg_candidate": True},
        ]
    }
    labels = [
        {"source": "sub-01/ses-01/eeg/sub-01_task-awake_run-01_eeg.edf", "state_label": "awake", "behavior_label": "responsive", "report_label": "experience", "task_label": "awake"},
        {"source": "sub-02/ses-01/eeg/sub-02_task-sedated_run-01_eeg.edf", "state_label": "sedated", "behavior_label": "unresponsive", "report_label": "no_experience", "task_label": "sedated"},
    ]
    (d / "file_inventory.json").write_text(json.dumps(inv), encoding="utf-8")
    (d / "label_candidates.json").write_text(json.dumps(labels), encoding="utf-8")
    (d / "contract_report.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    (d / "report.md").write_text("ok\n", encoding="utf-8")
    return d


def test_mock_fixture_windows_features():
    rows = extract_level_m_window_features(build_mock_level_m_windows_from_inspection())
    assert len({r.subject_id for r in rows}) >= 2
    assert len(rows) >= 2
    assert all(math.isfinite(r.spectral_power_proxy) for r in rows)


def test_missing_inspection_raises(tmp_path):
    missing = tmp_path / "x"
    try:
        load_bids_inspection_outputs(str(missing))
        assert False
    except FileNotFoundError:
        assert True


def test_build_from_minimal_inspection(tmp_path):
    d = _minimal_inspection(tmp_path)
    inspection = load_bids_inspection_outputs(str(d))
    rows = build_level_m_windows_from_bids_inventory(inspection)
    assert len(rows) == 4
    r = rows[0]
    assert r.row_id and r.subject_id and r.task_label is not None
    assert r.source_file
    assert r.window_start_s == 0.0 and r.window_end_s == 10.0


def test_task_mappings_and_guardrails():
    rows = extract_level_m_window_features(build_mock_level_m_windows_from_inspection())
    assert any(r.state_label == "sedated" and r.report_label == "experience" for r in rows)
    assert any(r.behavior_label == "unresponsive" and r.state_label != "unconscious" for r in rows)
    a = evaluate_level_m_windows(rows, "awake_vs_sedated")
    b = evaluate_level_m_windows(rows, "responsive_vs_unresponsive")
    c = evaluate_level_m_windows(rows, "experience_vs_no_experience")
    assert a.dataset_id == "ds005620" and a.task == "awake_vs_sedated" and a.n_rows > 0 and a.n_subjects >= 2
    for m in (a, b, c):
        assert m.auc is None or 0 <= m.auc <= 1
        assert m.brier is None or 0 <= m.brier <= 1
        assert m.ece is None or m.ece >= 0


def test_reports_and_duplicates(tmp_path):
    rows = extract_level_m_window_features(build_mock_level_m_windows_from_inspection())
    ar = build_window_artifact_report(rows)
    lr = build_window_leakage_report(rows)
    for k in ["mean_artifact_score", "max_artifact_score", "n_artifact_high", "artifact_dominance"]:
        assert k in ar
    for k in ["n_subjects", "subject_split_possible", "row_ids_unique", "leakage_detected"]:
        assert k in lr

    dup = [rows[0], LevelMWindowRow(**{**rows[0].__dict__})]
    assert build_window_leakage_report(dup)["leakage_detected"] is True

    out = tmp_path / "out"
    result = evaluate_level_m_windows(rows, "awake_vs_sedated")
    paths = write_level_m_window_outputs(result, str(out))
    assert set(paths.keys()) == {"features_m.csv", "metrics_m.json", "artifact_report.json", "leakage_report.json", "omega_event.json", "report.md"}
    for p in paths.values():
        assert (out / (p.split('/')[-1])).exists()
    parsed = json.loads((out / "metrics_m.json").read_text(encoding="utf-8"))
    assert "dataset_id" in parsed and "task" in parsed
    txt = (out / "report.md").read_text(encoding="utf-8").lower()
    for term in ["operational level m", "window-feature candidates", "future residual testing"]:
        assert term in txt
    for bad in ["proves consciousness", "soul proven", "afterlife proven", "liberation detected", "ontology solved", "ultimate reality", "q equals self", "q equals soul", "q_abs equals suffering", "f_dress equals karma"]:
        assert bad not in txt


def test_cli_mock_and_missing(tmp_path):
    out = tmp_path / "cli"
    r = subprocess.run([sys.executable, "-m", "sciencer_d.btc_icft.pipelines.run_ds005620_m_real", "--out", str(out), "--mock-fixture"], capture_output=True, text=True)
    assert r.returncode == 0
    for name in ["features_m.csv", "metrics_m.json", "artifact_report.json", "leakage_report.json", "omega_event.json", "report.md"]:
        assert (out / name).exists()

    miss = subprocess.run([sys.executable, "-m", "sciencer_d.btc_icft.pipelines.run_ds005620_m_real", "--inspection", str(tmp_path / "missing"), "--out", str(tmp_path / "o")], capture_output=True, text=True)
    assert miss.returncode != 0
    assert "Run inspect_ds005620_bids first or use --mock-fixture" in (miss.stdout + miss.stderr)


def test_config_contains_guardrails():
    txt = open("configs/btc_icft/ds005620_m_real.yaml", encoding="utf-8").read()
    for req in ["features_m.csv", "metrics_m.json", "artifact_report.json", "leakage_report.json", "omega_event.json", "report.md", "unresponsive_not_unconscious", "sedated_not_no_experience"]:
        assert req in txt
