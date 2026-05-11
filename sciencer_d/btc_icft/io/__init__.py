"""DS005620 EEG reader adapter contracts and signal-block contracts."""
from .eeg_readers import (
    EEGReaderCapability,
    EEGFileReadability,
    EEGChannelInventory,
    inspect_eeg_file,
    inspect_eeg_files,
    build_reader_capability_report,
    build_channel_inventory,
    write_eeg_reader_outputs,
)
from .eeg_signal_blocks import (
    EEGSignalFile,
    EEGSignalWindow,
    EEGSignalProbeResult,
    parse_fixture_signal_file,
    segment_signal_file,
    probe_signal_paths,
    build_signal_block_inventory,
    build_reader_alignment_report,
    write_signal_probe_outputs,
)

__all__ = [
    "EEGReaderCapability",
    "EEGFileReadability",
    "EEGChannelInventory",
    "inspect_eeg_file",
    "inspect_eeg_files",
    "build_reader_capability_report",
    "build_channel_inventory",
    "write_eeg_reader_outputs",
    "EEGSignalFile",
    "EEGSignalWindow",
    "EEGSignalProbeResult",
    "parse_fixture_signal_file",
    "segment_signal_file",
    "probe_signal_paths",
    "build_signal_block_inventory",
    "build_reader_alignment_report",
    "write_signal_probe_outputs",
]
