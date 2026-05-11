"""Tests for DS005620 EEG signal-block adapter contract (P8.2).

All tests are offline and stdlib-only. No MNE, scipy, or numpy required.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from sciencer_d.btc_icft.io.eeg_signal_blocks import (
    EEGSignalFile,
    EEGSignalWindow,
    EEGSignalProbeResult,
    parse_fixture_signal_file,
    segment_signal_file,
    probe_signal_paths,
    build_signal_block_inventory,
    build_reader_alignment_report,
    write_signal_probe_outputs,
    _validate_safe_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, n_channels: int = 4, n_rows: int = 100, sample_rate: float = 100.0) -> None:
    with open(path, "w") as f:
        f.write(f"# channels: {n_channels}\n")
        f.write(f"# sample_rate: {sample_rate}\n")
        headers = ["time"] + [f"ch{i+1}" for i in range(n_channels)]
        f.write(",".join(headers) + "\n")
        for i in range(n_rows):
            vals = [f"{i * 0.01:.4f}"] + [f"{i * 0.1 * (j+1):.4f}" for j in range(n_channels)]
            f.write(",".join(vals) + "\n")


def _write_txt(path: Path, n_channels: int = 3, n_rows: int = 50, sample_rate: float = 128.0) -> None:
    with open(path, "w") as f:
        f.write(f"# channels: {n_channels}\n")
        f.write(f"# sample_rate: {sample_rate}\n")
        for i in range(n_rows):
            f.write(" ".join([f"{i * 0.05 * (j+1):.4f}" for j in range(n_channels)]) + "\n")


def _write_tsv(path: Path, n_channels: int = 2, n_rows: int = 80) -> None:
    with open(path, "w") as f:
        f.write("# channels: 2\n")
        f.write("# sample_rate: 256.0\n")
        f.write("# channel_names: left,right\n")
        for i in range(n_rows):
            f.write(f"{i * 0.1:.4f}\t{-i * 0.1:.4f}\n")


# ---------------------------------------------------------------------------
# Class 1: EEGSignalFile dataclass
# ---------------------------------------------------------------------------

class TestEEGSignalFileDataclass:
    def test_default_fields(self):
        sf = EEGSignalFile(path="/tmp/x.csv", readable=False)
        assert sf.adapter is None
        assert sf.n_channels is None
        assert sf.n_samples is None
        assert sf.sample_rate_hz is None
        assert sf.duration_s is None
        assert sf.channel_names == []
        assert sf.samples == []
        assert sf.warnings == []
        assert sf.errors == []

    def test_readable_fields(self):
        sf = EEGSignalFile(
            path="/tmp/x.csv",
            readable=True,
            adapter="fixture_text",
            n_channels=4,
            n_samples=100,
            sample_rate_hz=250.0,
            duration_s=0.4,
            channel_names=["a", "b", "c", "d"],
            samples=[[0.1, 0.2, 0.3, 0.4]],
        )
        assert sf.readable is True
        assert sf.n_channels == 4
        assert sf.duration_s == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Class 2: parse_fixture_signal_file
# ---------------------------------------------------------------------------

class TestParseFixtureSignalFile:
    def test_parses_csv_header_and_data(self, tmp_path):
        p = tmp_path / "test.csv"
        _write_csv(p, n_channels=4, n_rows=100, sample_rate=250.0)
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is True
        assert sf.n_channels == 4
        assert sf.sample_rate_hz == pytest.approx(250.0)
        assert sf.n_samples == 100
        assert sf.duration_s == pytest.approx(0.4)

    def test_parses_txt_whitespace(self, tmp_path):
        p = tmp_path / "test.txt"
        _write_txt(p, n_channels=3, n_rows=50, sample_rate=128.0)
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is True
        assert sf.n_channels == 3
        assert sf.n_samples == 50

    def test_parses_tsv_with_channel_names(self, tmp_path):
        p = tmp_path / "test.tsv"
        _write_tsv(p, n_channels=2, n_rows=80)
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is True
        assert sf.n_channels == 2
        assert "left" in sf.channel_names or sf.channel_names[0] in {"left", "ch1"}

    def test_missing_file_returns_unreadable(self, tmp_path):
        sf = parse_fixture_signal_file(str(tmp_path / "nonexistent.csv"))
        assert sf.readable is False
        assert sf.errors

    def test_unsupported_extension_returns_unreadable(self, tmp_path):
        p = tmp_path / "test.edf"
        p.write_bytes(b"\x00\x01\x02")
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is False
        assert any("unsupported" in e.lower() for e in sf.errors)

    def test_empty_file_returns_unreadable(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is False

    def test_fallback_sample_rate_used_when_no_header(self, tmp_path):
        p = tmp_path / "nosrate.csv"
        with open(p, "w") as f:
            for i in range(20):
                f.write(f"{i * 0.1:.4f},{i * 0.2:.4f}\n")
        sf = parse_fixture_signal_file(str(p), sample_rate_hz=42.0)
        assert sf.readable is True
        assert sf.sample_rate_hz == pytest.approx(42.0)

    def test_auto_generates_channel_names(self, tmp_path):
        p = tmp_path / "nochnames.csv"
        with open(p, "w") as f:
            f.write("# channels: 3\n")
            f.write("# sample_rate: 100.0\n")
            for i in range(10):
                f.write(f"{i},{i*2},{i*3}\n")
        sf = parse_fixture_signal_file(str(p))
        assert sf.readable is True
        assert len(sf.channel_names) == 3
        assert all(n.startswith("ch") for n in sf.channel_names)


# ---------------------------------------------------------------------------
# Class 3: segment_signal_file
# ---------------------------------------------------------------------------

class TestSegmentSignalFile:
    def _make_signal_file(self, n_samples: int, sr: float = 100.0, n_ch: int = 4) -> EEGSignalFile:
        samples = [[float(i * j) for j in range(1, n_ch + 1)] for i in range(n_samples)]
        return EEGSignalFile(
            path="/tmp/mock.csv",
            readable=True,
            adapter="fixture_text",
            n_channels=n_ch,
            n_samples=n_samples,
            sample_rate_hz=sr,
            duration_s=n_samples / sr,
            channel_names=[f"ch{j+1}" for j in range(n_ch)],
            samples=samples,
        )

    def test_produces_correct_number_of_windows(self):
        sf = self._make_signal_file(n_samples=1000, sr=100.0)
        wins = segment_signal_file(sf, window_seconds=1.0, max_windows=None)
        assert len(wins) == 10  # 1000 samples / 100 = 10 full windows

    def test_row_id_format(self):
        sf = self._make_signal_file(n_samples=500, sr=100.0)
        wins = segment_signal_file(sf, window_seconds=1.0, max_windows=None)
        for idx, w in enumerate(wins[:5]):
            assert w.row_id == f"mock__win_{idx}"

    def test_max_windows_limits_output(self):
        sf = self._make_signal_file(n_samples=1000, sr=100.0)
        wins = segment_signal_file(sf, window_seconds=1.0, max_windows=3)
        assert len(wins) <= 3

    def test_short_file_produces_short_window(self):
        sf = self._make_signal_file(n_samples=10, sr=100.0)  # 0.1s < 10s window
        wins = segment_signal_file(sf, window_seconds=10.0)
        assert len(wins) == 1
        assert wins[0].status == "short_window"
        assert wins[0].n_samples == 10

    def test_unreadable_file_returns_empty(self):
        sf = EEGSignalFile(path="/tmp/bad.csv", readable=False)
        wins = segment_signal_file(sf, window_seconds=1.0)
        assert wins == []

    def test_window_timestamps_are_correct(self):
        sf = self._make_signal_file(n_samples=300, sr=100.0)
        wins = segment_signal_file(sf, window_seconds=1.0, max_windows=None)
        assert wins[0].window_start_s == pytest.approx(0.0)
        assert wins[0].window_end_s == pytest.approx(1.0)
        assert wins[1].window_start_s == pytest.approx(1.0)
        assert wins[1].window_end_s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Class 4: probe_signal_paths
# ---------------------------------------------------------------------------

class TestProbeSignalPaths:
    def test_probe_returns_correct_file_counts(self, tmp_path):
        csv_p = tmp_path / "a.csv"
        txt_p = tmp_path / "b.txt"
        _write_csv(csv_p, n_rows=200)
        _write_txt(txt_p, n_rows=50)
        result = probe_signal_paths([str(csv_p), str(txt_p)], sample_rate_hz=100.0)
        assert result.n_files == 2
        assert result.n_readable_files == 2

    def test_unsupported_extension_is_skipped(self, tmp_path):
        edf_p = tmp_path / "test.edf"
        edf_p.write_bytes(b"\x00\x01")
        csv_p = tmp_path / "test.csv"
        _write_csv(csv_p, n_rows=100)
        result = probe_signal_paths([str(edf_p), str(csv_p)], sample_rate_hz=100.0)
        assert result.n_skipped_files == 1
        assert result.n_readable_files == 1

    def test_safe_claim_present_and_no_banned_phrases(self, tmp_path):
        csv_p = tmp_path / "x.csv"
        _write_csv(csv_p, n_rows=50)
        result = probe_signal_paths([str(csv_p)])
        assert result.safe_claim
        _validate_safe_text(result.safe_claim)

    def test_windows_generated_from_readable_files(self, tmp_path):
        csv_p = tmp_path / "w.csv"
        _write_csv(csv_p, n_rows=500, sample_rate=100.0)
        result = probe_signal_paths([str(csv_p)], window_seconds=1.0, max_windows_per_file=3)
        assert result.n_windows > 0
        assert result.n_windows <= 3


# ---------------------------------------------------------------------------
# Class 5: write_signal_probe_outputs
# ---------------------------------------------------------------------------

class TestWriteSignalProbeOutputs:
    def _make_result(self, tmp_path: Path) -> EEGSignalProbeResult:
        csv_p = tmp_path / "sig.csv"
        _write_csv(csv_p, n_channels=4, n_rows=200, sample_rate=100.0)
        return probe_signal_paths([str(csv_p)], window_seconds=1.0, max_windows_per_file=3)

    def test_writes_all_six_artifacts(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = str(tmp_path / "out")
        outputs = write_signal_probe_outputs(result, out_dir)
        required = {
            "signal_block_inventory",
            "window_inventory",
            "reader_alignment_report",
            "skipped_files",
            "omega_event",
            "report",
        }
        assert required == set(outputs.keys())
        for name, path in outputs.items():
            assert Path(path).exists(), f"Missing artifact: {name}"

    def test_signal_block_inventory_json_valid(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = str(tmp_path / "out")
        outputs = write_signal_probe_outputs(result, out_dir)
        with open(outputs["signal_block_inventory"]) as f:
            data = json.load(f)
        assert "n_files" in data
        assert "n_windows" in data
        assert "files" in data

    def test_window_inventory_csv_has_header(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = str(tmp_path / "out")
        outputs = write_signal_probe_outputs(result, out_dir)
        with open(outputs["window_inventory"], newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "row_id" in header
        assert "window_start_s" in header
        assert "status" in header

    def test_omega_event_has_event_id(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = str(tmp_path / "out")
        outputs = write_signal_probe_outputs(result, out_dir)
        with open(outputs["omega_event"]) as f:
            data = json.load(f)
        assert "event_id" in data
        assert len(data["event_id"]) == 16

    def test_report_md_contains_overview(self, tmp_path):
        result = self._make_result(tmp_path)
        out_dir = str(tmp_path / "out")
        outputs = write_signal_probe_outputs(result, out_dir)
        report_text = Path(outputs["report"]).read_text()
        assert "Signal-Block" in report_text
        assert "Guardrail" in report_text


# ---------------------------------------------------------------------------
# Class 6: Guardrails and contract constraints
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_validate_safe_text_passes_clean_text(self):
        _validate_safe_text("This is a numeric signal block for exploratory analysis.")

    def test_validate_safe_text_raises_on_banned_phrase(self):
        with pytest.raises(ValueError, match="consciousness"):
            _validate_safe_text("This probe measures consciousness states.")

    def test_validate_safe_text_raises_on_soul(self):
        with pytest.raises(ValueError, match="soul"):
            _validate_safe_text("The soul of the signal.")

    def test_no_samples_stored_in_window_objects(self, tmp_path):
        csv_p = tmp_path / "g.csv"
        _write_csv(csv_p, n_rows=200, sample_rate=100.0)
        result = probe_signal_paths([str(csv_p)], window_seconds=1.0, max_windows_per_file=2)
        for w in result.windows:
            # EEGSignalWindow is metadata-only; it must NOT have a 'samples' attribute
            assert not hasattr(w, "samples")

    def test_row_id_determinism(self, tmp_path):
        csv_p = tmp_path / "det.csv"
        _write_csv(csv_p, n_rows=200, sample_rate=100.0)
        result1 = probe_signal_paths([str(csv_p)], window_seconds=1.0, max_windows_per_file=5)
        result2 = probe_signal_paths([str(csv_p)], window_seconds=1.0, max_windows_per_file=5)
        ids1 = [w.row_id for w in result1.windows]
        ids2 = [w.row_id for w in result2.windows]
        assert ids1 == ids2

    def test_forbidden_claims_list_is_empty_by_default(self, tmp_path):
        csv_p = tmp_path / "fc.csv"
        _write_csv(csv_p, n_rows=50)
        result = probe_signal_paths([str(csv_p)])
        assert result.forbidden_claims == []
