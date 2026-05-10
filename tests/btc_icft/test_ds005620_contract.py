import json
from pathlib import Path

from sciencer_d.btc_icft.datasets.ds005620 import (
    DS005620DatasetConfig,
    DS005620LabelRow,
    build_ds005620_dataset_card,
    validate_ds005620_contract,
    validate_ds005620_label_row,
    write_ds005620_contract_outputs,
)


def _config(subject_split_required=True, artifact_report_required=True):
    return DS005620DatasetConfig(
        required_tasks=[
            "awake_vs_sedated",
            "responsive_vs_unresponsive",
            "experience_vs_no_experience",
            "loc_roc_transition",
        ],
        allowed_state_labels=["awake", "sedated", "loc_candidate", "roc_candidate", "unknown"],
        allowed_behavior_labels=["responsive", "unresponsive", "partial", "unknown"],
        allowed_report_labels=["experience", "no_experience", "dream", "imagery", "discontinuity", "unknown"],
        notes=["operational labels only", "label contract for residual testing"],
        subject_split_required=subject_split_required,
        artifact_report_required=artifact_report_required,
    )


def _valid_rows():
    return [
        DS005620LabelRow(row_id="r1", subject_id="sub-01", state_label="awake", behavior_label="responsive", report_label="experience", task_label="awake_vs_sedated", confidence=0.9),
        DS005620LabelRow(row_id="r2", subject_id="sub-02", state_label="sedated", behavior_label="unresponsive", report_label="no_experience", task_label="responsive_vs_unresponsive", confidence=0.7),
    ]


def test_valid_minimal_contract_passes():
    report = validate_ds005620_contract(_valid_rows(), _config())
    assert report.valid is True


def test_missing_subject_id_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r1", subject_id=""))
    assert any("subject_id is required" in e for e in errs)


def test_missing_row_id_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="", subject_id="sub-01"))
    assert any("row_id is required" in e for e in errs)


def test_confidence_out_of_range_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", confidence=1.2))
    assert any("confidence" in e for e in errs)


def test_unresponsive_implying_unconscious_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", behavior_label="unresponsive", notes=["this is unconscious"] ))
    assert any("equated" in e for e in errs)


def test_sedated_means_no_experience_phrase_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", state_label="sedated", report_label="no_experience", notes=["sedated means no_experience"]))
    assert any("sedated" in e for e in errs)


def test_behavior_label_no_experience_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", behavior_label="no_experience"))
    assert errs


def test_state_label_unconscious_fails():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", state_label="unconscious"))
    assert errs


def test_report_labels_as_state_fail():
    for lbl in ["experience", "no_experience"]:
        errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", state_label=lbl))
        assert errs


def test_forbidden_terms_fail():
    errs = validate_ds005620_label_row(DS005620LabelRow(row_id="r", subject_id="sub-01", task_label="ultimate reality", notes=["soul afterlife liberation enlightenment ontology solved"]))
    assert len(errs) >= 1


def test_subject_split_requirement_fails_with_single_subject():
    rows = [DS005620LabelRow(row_id="r1", subject_id="sub-01")]
    report = validate_ds005620_contract(rows, _config(subject_split_required=True))
    assert report.valid is False


def test_required_outputs_include_artifact_report():
    report = validate_ds005620_contract(_valid_rows(), _config(artifact_report_required=True))
    assert "artifact_report.json" in report.required_outputs


def test_dataset_card_counts_correctly():
    card = build_ds005620_dataset_card(_config(), _valid_rows())
    assert card.n_subjects == 2
    assert card.state_labels["awake"] == 1


def test_write_outputs_and_json_keys(tmp_path):
    out = write_ds005620_contract_outputs(_config(), _valid_rows(), str(tmp_path))
    assert Path(out["dataset_card.json"]).exists()
    assert Path(out["label_contract_report.json"]).exists()
    assert Path(out["report.md"]).exists()

    card = json.loads(Path(out["dataset_card.json"]).read_text(encoding="utf-8"))
    report = json.loads(Path(out["label_contract_report.json"]).read_text(encoding="utf-8"))
    assert "dataset_id" in card
    assert "valid" in report


def test_report_contains_cautious_terms_and_no_banned(tmp_path):
    out = write_ds005620_contract_outputs(_config(), _valid_rows(), str(tmp_path))
    txt = Path(out["report.md"]).read_text(encoding="utf-8").lower()
    assert "operational" in txt
    assert "label contract" in txt
    assert "residual testing" in txt
    banned = [
        "proves consciousness",
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
    for phrase in banned:
        assert phrase not in txt


def test_config_file_exists_and_has_required_tasks():
    text = Path("configs/btc_icft/ds005620.yaml").read_text(encoding="utf-8")
    assert "awake_vs_sedated" in text
    assert "responsive_vs_unresponsive" in text
    assert "experience_vs_no_experience" in text
    assert "loc_roc_transition" in text
