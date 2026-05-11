from __future__ import annotations

import json
import subprocess
import sys

import yaml

from sciencer_d.btc_icft.datasets.eeg_study_registry import (
    discover_study_files,
    get_study_record,
    get_study_registry,
    inspect_study_dataset,
    write_study_dataset_outputs,
)


BANNED = [
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
]


def test_registry_contains_all_seed_studies():
    registry = get_study_registry()
    for dataset_id in ["DS005620", "DS002094", "ds001787", "ds003969", "ds003816", "PhysioNet_GABA"]:
        assert dataset_id in registry


def test_unknown_dataset_id_raises_value_error():
    import pytest
    with pytest.raises(ValueError):
        get_study_record("unknown")


def test_mock_fixture_mode_writes_all_outputs(tmp_path):
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "sciencer_d.btc_icft.pipelines.feed_eeg_study_dataset", "--dataset-id", "DS005620", "--mock-fixture", "--out", str(out)]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0
    for name in ["study_card.json", "file_readability_report.json", "reader_capability_report.json", "dataset_readiness_report.json", "report.md"]:
        assert (out / name).exists()


def test_fixture_files_become_fixture_or_ready(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.csv").write_text("# channels: 2\n# sample_rate: 128\n1,2\n")
    (root / "b.txt").write_text("# channels: 1\n# sample_rate: 64\n1\n")
    (root / "c.tsv").write_text("# channels: 2\n# sample_rate: 128\n1\t2\n")
    res = inspect_study_dataset("DS005620", str(root))
    assert res.readiness_status in {"fixture_readable", "ready_for_p9_signal_extraction"}


def test_unsupported_edf_without_mne_reports_and_no_crash(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.edf").write_text("fake")
    res = inspect_study_dataset("DS005620", str(root))
    assert res.readiness_status == "unsupported_or_dependency_missing"


def test_missing_root_fails_cleanly(tmp_path):
    missing = tmp_path / "missing"
    res = inspect_study_dataset("DS005620", str(missing))
    assert res.readiness_status == "missing_root"


def test_output_json_parses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.csv").write_text("# channels: 2\n# sample_rate: 128\n1,2\n")
    res = inspect_study_dataset("DS005620", str(root))
    out = write_study_dataset_outputs(res, str(tmp_path / "out"))
    for key in ["study_card", "file_readability_report", "reader_capability_report", "dataset_readiness_report"]:
        json.loads((tmp_path / "out" / (key + ".json" if key != "study_card" else "study_card.json")).read_text())


def test_report_avoids_banned_phrases(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.csv").write_text("# channels: 2\n# sample_rate: 128\n1,2\n")
    res = inspect_study_dataset("DS005620", str(root))
    write_study_dataset_outputs(res, str(tmp_path / "out"))
    text = (tmp_path / "out" / "report.md").read_text().lower()
    for phrase in BANNED:
        assert phrase not in text


def test_cli_mock_fixture_returns_zero_and_writes_outputs(tmp_path):
    out = tmp_path / "out"
    cp = subprocess.run([sys.executable, "-m", "sciencer_d.btc_icft.pipelines.feed_eeg_study_dataset", "--dataset-id", "DS005620", "--mock-fixture", "--out", str(out)], capture_output=True, text=True)
    assert cp.returncode == 0
    assert (out / "study_card.json").exists()
    assert (out / "file_readability_report.json").exists()
    assert (out / "reader_capability_report.json").exists()
    assert (out / "dataset_readiness_report.json").exists()
    assert (out / "report.md").exists()


def test_config_contains_study_ids_outputs_guardrails():
    cfg = yaml.safe_load(open("configs/btc_icft/eeg_studies.yaml", "r"))
    assert set(["DS005620", "DS002094", "ds001787", "ds003969", "ds003816", "PhysioNet_GABA"]).issubset(set(cfg["study_ids"]))
    assert "study_card.json" in cfg["required_outputs"]
    assert "no_data_download" in cfg["guardrails"]
