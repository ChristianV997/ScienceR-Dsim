"""DS005620 EEG signal-block adapter contract.

Turns fixture EEG-like files (CSV/TSV/TXT) into windowed numeric signal blocks.

Does NOT:
- download data
- add dependencies (stdlib-only)
- extract Level M features
- perform Level T topology
- train models
- make consciousness/soul/liberation/afterlife/ontology claims
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import hashlib


_BANNED_PHRASES = [
    "consciousness",
    "soul",
    "liberation",
    "afterlife",
    "sentient",
    "enlightenment",
    "awakening",
    "nirvana",
    "transcendence",
]


def _validate_safe_text(text: str) -> None:
    """Raise ValueError if text contains any banned phrase.

    Args:
        text: Text to validate.

    Raises:
        ValueError: If any banned phrase is found.
    """
    lower = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            raise ValueError(f"Banned phrase detected in output text: {phrase!r}")


@dataclass
class EEGSignalFile:
    """Parsed numeric signal from a single EEG fixture file."""
    path: str
    readable: bool
    adapter: Optional[str] = None
    n_channels: Optional[int] = None
    n_samples: Optional[int] = None
    sample_rate_hz: Optional[float] = None
    duration_s: Optional[float] = None
    channel_names: list[str] = field(default_factory=list)
    samples: list[list[float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class EEGSignalWindow:
    """Metadata-only descriptor for a single windowed signal block."""
    file_path: str
    row_id: str
    window_id: int
    window_start_s: float
    window_end_s: float
    sample_start: int
    sample_end: int
    n_channels: int
    n_samples: int
    sample_rate_hz: float
    channel_names: list[str] = field(default_factory=list)
    status: str = "ok"
    warnings: list[str] = field(default_factory=list)


@dataclass
class EEGSignalProbeResult:
    """Aggregate result from probing a set of EEG fixture files."""
    n_files: int
    n_readable_files: int
    n_skipped_files: int
    n_windows: int
    signal_files: list[EEGSignalFile] = field(default_factory=list)
    windows: list[EEGSignalWindow] = field(default_factory=list)
    skipped_files: list[dict] = field(default_factory=list)
    reader_alignment_report: dict = field(default_factory=dict)
    safe_claim: str = ""
    forbidden_claims: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_fixture_signal_file(
    path: str,
    sample_rate_hz: float = 100.0,
) -> EEGSignalFile:
    """Parse a fixture text file (CSV/TSV/TXT) into a numeric signal.

    Reads header comments of the form:
        # channels: 8
        # sample_rate: 250.0
        # channel_names: ch1,ch2,...

    Then reads numeric data rows. The first non-comment, non-header row
    that contains only numbers becomes a data row. A header row with column
    names is also detected and used for channel_names if not already set.

    Args:
        path: Path to the fixture file.
        sample_rate_hz: Fallback sample rate if not found in file header.

    Returns:
        EEGSignalFile with parsed data.
    """
    p = Path(path)
    result = EEGSignalFile(
        path=str(p),
        readable=False,
        adapter="fixture_text",
    )

    if not p.exists():
        result.errors.append(f"File not found: {path}")
        return result

    ext = p.suffix.lower()
    if ext not in {".csv", ".tsv", ".txt"}:
        result.errors.append(f"Unsupported extension for fixture adapter: {ext}")
        return result

    try:
        with open(p, "r", newline="") as f:
            raw = f.read()
    except Exception as e:
        result.errors.append(f"Could not read file: {e}")
        return result

    lines = raw.splitlines()

    n_channels_header: Optional[int] = None
    sr_header: Optional[float] = None
    channel_names_header: Optional[list[str]] = None
    data_rows: list[list[float]] = []
    detected_header_row: Optional[list[str]] = None
    errors: list[str] = []
    warnings: list[str] = []

    # Determine delimiter
    delimiter = ","
    if ext == ".tsv":
        delimiter = "\t"
    elif ext == ".txt":
        # Probe first non-comment line
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "\t" in stripped:
                delimiter = "\t"
            elif "," in stripped:
                delimiter = ","
            else:
                delimiter = None  # whitespace-split
            break

    def _split(line: str) -> list[str]:
        if delimiter is None:
            return line.split()
        return [x.strip() for x in line.split(delimiter)]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Parse header comments
        if stripped.startswith("#"):
            content = stripped[1:].strip()
            if content.startswith("channels:") or content.startswith("channels :"):
                try:
                    n_channels_header = int(content.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    warnings.append(f"Line {i+1}: malformed channel count comment")
            elif content.startswith("sample_rate:") or content.startswith("sample_rate :"):
                try:
                    sr_header = float(content.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    warnings.append(f"Line {i+1}: malformed sample_rate comment")
            elif content.startswith("channel_names:") or content.startswith("channel_names :"):
                try:
                    names_str = content.split(":", 1)[1].strip()
                    channel_names_header = [n.strip() for n in names_str.split(",")]
                except (ValueError, IndexError):
                    warnings.append(f"Line {i+1}: malformed channel_names comment")
            continue

        # Try to parse as numeric row
        parts = _split(stripped)
        if not parts:
            continue

        try:
            nums = [float(x) for x in parts]
            data_rows.append(nums)
        except ValueError:
            # Non-numeric: could be a header row with column names
            if detected_header_row is None and all(not _is_numeric(x) for x in parts):
                detected_header_row = parts
            # Skip non-numeric, non-header rows silently

    # Determine channel names
    channel_names: list[str] = []
    if channel_names_header:
        channel_names = channel_names_header
    elif detected_header_row:
        # Filter out a leading time/timestamp column
        if detected_header_row and detected_header_row[0].lower() in {"time", "t", "timestamp"}:
            channel_names = detected_header_row[1:]
        else:
            channel_names = detected_header_row

    # Normalize data rows (strip leading time column if widths match channel count + 1)
    if data_rows:
        first_row_width = len(data_rows[0])

        # If header names suggest first col is time
        has_time_col = (
            detected_header_row
            and detected_header_row[0].lower() in {"time", "t", "timestamp"}
            and first_row_width == len(detected_header_row)
        )

        if has_time_col:
            data_rows = [row[1:] for row in data_rows]
            first_row_width = len(data_rows[0]) if data_rows else 0

        # Validate consistent width
        inconsistent = [
            j for j, row in enumerate(data_rows) if len(row) != first_row_width
        ]
        if inconsistent:
            warnings.append(
                f"{len(inconsistent)} rows have inconsistent column count (expected {first_row_width})"
            )
            data_rows = [r for r in data_rows if len(r) == first_row_width]

    # Determine final metadata
    n_ch_final: Optional[int] = None
    if n_channels_header is not None:
        n_ch_final = n_channels_header
    elif data_rows:
        n_ch_final = len(data_rows[0])

    sr_final = sr_header if sr_header is not None else sample_rate_hz

    # Generate channel names if missing
    if n_ch_final and not channel_names:
        channel_names = [f"ch{j+1}" for j in range(n_ch_final)]

    n_samples = len(data_rows)
    duration_s = n_samples / sr_final if sr_final and n_samples else None

    result.readable = n_samples > 0
    result.n_channels = n_ch_final
    result.n_samples = n_samples
    result.sample_rate_hz = sr_final
    result.duration_s = duration_s
    result.channel_names = channel_names[:n_ch_final] if n_ch_final and channel_names else channel_names
    result.samples = data_rows
    result.warnings = warnings
    result.errors = errors

    if not result.readable:
        result.errors.append("No numeric data rows found")

    return result


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def segment_signal_file(
    signal_file: EEGSignalFile,
    window_seconds: float = 10.0,
    max_windows: Optional[int] = None,
) -> list[EEGSignalWindow]:
    """Segment a parsed signal file into fixed-length windows.

    Args:
        signal_file: Parsed EEGSignalFile.
        window_seconds: Duration of each window in seconds.
        max_windows: Maximum number of windows to produce. None = no limit.

    Returns:
        List of EEGSignalWindow descriptors (metadata only, no raw samples).
    """
    windows: list[EEGSignalWindow] = []

    if not signal_file.readable or not signal_file.n_samples or not signal_file.sample_rate_hz:
        return windows

    stem = Path(signal_file.path).stem
    sr = signal_file.sample_rate_hz
    n_total = signal_file.n_samples
    n_ch = signal_file.n_channels or 0
    ch_names = signal_file.channel_names

    window_samples = int(math.floor(window_seconds * sr))
    if window_samples <= 0:
        window_samples = 1

    n_full_windows = n_total // window_samples
    has_remainder = (n_total % window_samples) > 0

    # If no full windows but file has samples → emit one short window
    if n_full_windows == 0 and n_total > 0:
        row_id = f"{stem}__win_0"
        w = EEGSignalWindow(
            file_path=signal_file.path,
            row_id=row_id,
            window_id=0,
            window_start_s=0.0,
            window_end_s=n_total / sr,
            sample_start=0,
            sample_end=n_total,
            n_channels=n_ch,
            n_samples=n_total,
            sample_rate_hz=sr,
            channel_names=ch_names,
            status="short_window",
            warnings=[f"Window shorter than requested ({n_total} samples < {window_samples})"],
        )
        windows.append(w)
        return windows

    limit = n_full_windows
    if max_windows is not None:
        limit = min(limit, max_windows)

    for idx in range(limit):
        sample_start = idx * window_samples
        sample_end = sample_start + window_samples
        start_s = sample_start / sr
        end_s = sample_end / sr
        row_id = f"{stem}__win_{idx}"
        w = EEGSignalWindow(
            file_path=signal_file.path,
            row_id=row_id,
            window_id=idx,
            window_start_s=start_s,
            window_end_s=end_s,
            sample_start=sample_start,
            sample_end=sample_end,
            n_channels=n_ch,
            n_samples=window_samples,
            sample_rate_hz=sr,
            channel_names=ch_names,
            status="ok",
        )
        windows.append(w)

    # Emit a short tail window if remainder exists and under max_windows limit
    if has_remainder and (max_windows is None or len(windows) < max_windows):
        idx = n_full_windows
        sample_start = idx * window_samples
        sample_end = n_total
        tail_samples = sample_end - sample_start
        start_s = sample_start / sr
        end_s = sample_end / sr
        row_id = f"{stem}__win_{idx}"
        w = EEGSignalWindow(
            file_path=signal_file.path,
            row_id=row_id,
            window_id=idx,
            window_start_s=start_s,
            window_end_s=end_s,
            sample_start=sample_start,
            sample_end=sample_end,
            n_channels=n_ch,
            n_samples=tail_samples,
            sample_rate_hz=sr,
            channel_names=ch_names,
            status="short_window",
            warnings=[f"Tail window shorter than requested ({tail_samples} samples < {window_samples})"],
        )
        windows.append(w)

    return windows


def probe_signal_paths(
    paths: list[str],
    sample_rate_hz: float = 100.0,
    window_seconds: float = 10.0,
    max_windows_per_file: Optional[int] = 3,
) -> EEGSignalProbeResult:
    """Probe a list of file paths for signal block extraction.

    Parses each file with the fixture_text adapter, segments into windows,
    and assembles an aggregate probe result.

    Args:
        paths: List of file paths to probe.
        sample_rate_hz: Fallback sample rate for files without header info.
        window_seconds: Window duration in seconds.
        max_windows_per_file: Max windows per file. None = no limit.

    Returns:
        EEGSignalProbeResult with all signal files, windows, and metadata.
    """
    signal_files: list[EEGSignalFile] = []
    all_windows: list[EEGSignalWindow] = []
    skipped: list[dict] = []
    warnings: list[str] = []

    _FIXTURE_EXTENSIONS = {".csv", ".tsv", ".txt"}

    for p in paths:
        path_obj = Path(p)
        ext = path_obj.suffix.lower()

        if ext not in _FIXTURE_EXTENSIONS:
            skipped.append({
                "path": str(p),
                "reason": f"unsupported_extension:{ext}",
            })
            warnings.append(f"Skipped {path_obj.name}: unsupported extension {ext!r}")
            continue

        sig = parse_fixture_signal_file(str(p), sample_rate_hz=sample_rate_hz)
        signal_files.append(sig)

        if not sig.readable:
            skipped.append({
                "path": str(p),
                "reason": "unreadable",
                "errors": sig.errors,
            })
            continue

        wins = segment_signal_file(sig, window_seconds=window_seconds, max_windows=max_windows_per_file)
        all_windows.extend(wins)

    n_readable = sum(1 for sf in signal_files if sf.readable)
    n_skipped = len(skipped)

    safe_claim = (
        "Signal blocks are numeric fixture arrays segmented for exploratory analysis. "
        "They are NOT validated biomarkers and carry no causal or diagnostic claims."
    )
    _validate_safe_text(safe_claim)

    reader_alignment = build_reader_alignment_report_from_parts(signal_files, skipped)

    result = EEGSignalProbeResult(
        n_files=len(paths),
        n_readable_files=n_readable,
        n_skipped_files=n_skipped,
        n_windows=len(all_windows),
        signal_files=signal_files,
        windows=all_windows,
        skipped_files=skipped,
        reader_alignment_report=reader_alignment,
        safe_claim=safe_claim,
        forbidden_claims=[],
        warnings=warnings,
    )
    return result


def build_signal_block_inventory(result: EEGSignalProbeResult) -> dict:
    """Build a JSON-serialisable inventory of signal blocks.

    Args:
        result: Aggregate probe result.

    Returns:
        Dict with aggregate counts and per-file summaries.
    """
    per_file = []
    for sf in result.signal_files:
        per_file.append({
            "path": sf.path,
            "readable": sf.readable,
            "adapter": sf.adapter,
            "n_channels": sf.n_channels,
            "n_samples": sf.n_samples,
            "sample_rate_hz": sf.sample_rate_hz,
            "duration_s": sf.duration_s,
            "channel_names": sf.channel_names,
            "n_warnings": len(sf.warnings),
            "n_errors": len(sf.errors),
        })

    windows_per_file: dict[str, int] = {}
    for w in result.windows:
        windows_per_file[w.file_path] = windows_per_file.get(w.file_path, 0) + 1

    return {
        "n_files": result.n_files,
        "n_readable_files": result.n_readable_files,
        "n_skipped_files": result.n_skipped_files,
        "n_windows": result.n_windows,
        "files": per_file,
        "windows_per_file": windows_per_file,
        "safe_claim": result.safe_claim,
    }


def build_reader_alignment_report_from_parts(
    signal_files: list[EEGSignalFile],
    skipped: list[dict],
) -> dict:
    """Build reader alignment report from parsed signal files and skipped list.

    Args:
        signal_files: List of parsed signal files.
        skipped: List of skipped-file dicts.

    Returns:
        Dict with alignment statistics.
    """
    adapters_used: dict[str, int] = {}
    extensions_seen: dict[str, int] = {}
    readable_count = 0

    for sf in signal_files:
        ext = Path(sf.path).suffix.lower()
        extensions_seen[ext] = extensions_seen.get(ext, 0) + 1
        if sf.adapter:
            adapters_used[sf.adapter] = adapters_used.get(sf.adapter, 0) + 1
        if sf.readable:
            readable_count += 1

    for item in skipped:
        ext = Path(item["path"]).suffix.lower()
        extensions_seen[ext] = extensions_seen.get(ext, 0) + 1

    return {
        "adapters_used": adapters_used,
        "extensions_seen": extensions_seen,
        "n_readable": readable_count,
        "n_skipped": len(skipped),
        "alignment": "fixture_text_only",
        "notes": [
            "Only fixture text extensions (.csv, .tsv, .txt) are processed.",
            "Binary EEG formats (.edf, .bdf, etc.) are not supported in this adapter.",
        ],
    }


def build_reader_alignment_report(result: EEGSignalProbeResult) -> dict:
    """Build reader alignment report from a probe result.

    Args:
        result: Aggregate probe result.

    Returns:
        Dict with alignment statistics.
    """
    return result.reader_alignment_report


def _build_omega_event(result: EEGSignalProbeResult) -> dict:
    """Build a minimal omega event record for the probe run.

    Args:
        result: Aggregate probe result.

    Returns:
        Dict with omega event metadata.
    """
    payload = f"probe:{result.n_files}:{result.n_windows}:{result.safe_claim}"
    event_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return {
        "event_id": event_id,
        "event_type": "signal_block_probe",
        "n_files": result.n_files,
        "n_readable_files": result.n_readable_files,
        "n_windows": result.n_windows,
        "safe_claim": result.safe_claim,
        "forbidden_claims": result.forbidden_claims,
        "warnings": result.warnings[:20],
    }


def _build_markdown_report(result: EEGSignalProbeResult) -> str:
    """Build a Markdown summary report for the probe run.

    Args:
        result: Aggregate probe result.

    Returns:
        Markdown string.
    """
    md = "# DS005620 EEG Signal-Block Adapter Inspection\n\n"
    md += "## Overview\n\n"
    md += "This is a **signal-block contract inspection**. It does NOT:\n"
    md += "- extract Level M features\n"
    md += "- perform Level T topology\n"
    md += "- train models or compute scientific evidence\n"
    md += "- make claims about cognition, affect, or physiology\n\n"

    md += f"**Safe claim:** {result.safe_claim}\n\n"

    md += "## Summary\n\n"
    md += f"- Total files probed: {result.n_files}\n"
    md += f"- Readable files: {result.n_readable_files}\n"
    md += f"- Skipped files: {result.n_skipped_files}\n"
    md += f"- Total windows: {result.n_windows}\n\n"

    if result.signal_files:
        md += "## File Details\n\n"
        for sf in result.signal_files[:20]:
            icon = "✓" if sf.readable else "✗"
            md += f"- {icon} {Path(sf.path).name}\n"
            if sf.n_channels:
                md += f"  - Channels: {sf.n_channels}\n"
            if sf.sample_rate_hz:
                md += f"  - Sample rate: {sf.sample_rate_hz} Hz\n"
            if sf.duration_s is not None:
                md += f"  - Duration: {sf.duration_s:.2f} s\n"
            if sf.errors:
                for e in sf.errors[:3]:
                    md += f"  - Error: {e}\n"

    if result.windows:
        md += "\n## Window Details\n\n"
        md += "| row_id | start_s | end_s | n_samples | status |\n"
        md += "|--------|---------|-------|-----------|--------|\n"
        for w in result.windows[:30]:
            md += f"| {w.row_id} | {w.window_start_s:.2f} | {w.window_end_s:.2f} | {w.n_samples} | {w.status} |\n"

    if result.skipped_files:
        md += "\n## Skipped Files\n\n"
        for item in result.skipped_files[:10]:
            md += f"- {Path(item['path']).name}: {item['reason']}\n"

    if result.warnings:
        md += f"\n## Warnings ({len(result.warnings)})\n\n"
        for warn in result.warnings[:10]:
            md += f"- {warn}\n"

    md += "\n---\n"
    md += "**Guardrail:** This inspection does not validate cognition, affect, or physiology.\n"
    return md


def write_signal_probe_outputs(
    result: EEGSignalProbeResult,
    out_dir: str,
) -> dict[str, str]:
    """Write signal probe outputs to files.

    Artifacts written:
        signal_block_inventory.json
        window_inventory.csv
        reader_alignment_report.json
        skipped_files.json
        omega_event.json
        report.md

    Args:
        result: Aggregate probe result.
        out_dir: Output directory path.

    Returns:
        Dict mapping artifact names to file paths written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # 1. signal_block_inventory.json
    inventory = build_signal_block_inventory(result)
    inv_file = out_path / "signal_block_inventory.json"
    with open(inv_file, "w") as f:
        json.dump(inventory, f, indent=2)
    outputs["signal_block_inventory"] = str(inv_file)

    # 2. window_inventory.csv
    win_file = out_path / "window_inventory.csv"
    with open(win_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_id", "file_path", "window_id",
            "window_start_s", "window_end_s",
            "sample_start", "sample_end",
            "n_channels", "n_samples", "sample_rate_hz", "status",
        ])
        for w in result.windows:
            writer.writerow([
                w.row_id, w.file_path, w.window_id,
                w.window_start_s, w.window_end_s,
                w.sample_start, w.sample_end,
                w.n_channels, w.n_samples, w.sample_rate_hz, w.status,
            ])
    outputs["window_inventory"] = str(win_file)

    # 3. reader_alignment_report.json
    align_file = out_path / "reader_alignment_report.json"
    with open(align_file, "w") as f:
        json.dump(result.reader_alignment_report, f, indent=2)
    outputs["reader_alignment_report"] = str(align_file)

    # 4. skipped_files.json
    skip_file = out_path / "skipped_files.json"
    with open(skip_file, "w") as f:
        json.dump({"skipped": result.skipped_files}, f, indent=2)
    outputs["skipped_files"] = str(skip_file)

    # 5. omega_event.json
    omega = _build_omega_event(result)
    omega_file = out_path / "omega_event.json"
    with open(omega_file, "w") as f:
        json.dump(omega, f, indent=2)
    outputs["omega_event"] = str(omega_file)

    # 6. report.md
    md = _build_markdown_report(result)
    md_file = out_path / "report.md"
    with open(md_file, "w") as f:
        f.write(md)
    outputs["report"] = str(md_file)

    return outputs
