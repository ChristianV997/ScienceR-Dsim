from __future__ import annotations

import json
import sqlite3

import pandas as pd
import yaml

from external.ecosystem import build_default_registry, extract_topology_metrics
from pipelines.run_external import run


def test_default_registry_contains_required_protocols():
    reg = build_default_registry()
    assert {"file", "rest", "mqtt", "websocket"}.issubset(reg.kinds)


def test_extract_topology_metrics_from_phase_deltas():
    row = extract_topology_metrics({"phase_deltas": [3.1415926535, -3.1415926535]})
    assert "Q" in row and "Qabs" in row and "f_dress" in row
    assert row["Qabs"] >= 0


def test_run_external_file_connector_and_db_logging(tmp_path):
    sensor_path = tmp_path / "live.jsonl"
    records = [
        {"timestamp": 1.0, "sensor_id": "s1", "Q": 1.0, "Qabs": 1.0},
        {"timestamp": 2.0, "sensor_id": "s1", "phase_deltas": [1.0, -0.5, 0.5]},
    ]
    sensor_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"external": {"connectors": [{"type": "file", "params": {"path": str(sensor_path)}}]}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out_csv = tmp_path / "live.csv"
    db_path = tmp_path / "runs.sqlite"
    df = run(cfg_path, out_csv, db_path, max_records=10)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert {"timestamp", "sensor_id", "protocol", "Q", "Qabs", "f_dress"}.issubset(df.columns)
    assert out_csv.exists()
    assert (tmp_path / "live.csv.meta.json").exists()

    conn = sqlite3.connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
        assert n == 2
    finally:
        conn.close()
