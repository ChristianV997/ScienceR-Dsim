from __future__ import annotations
import pytest
from database.database import connect, start_run, finish_run, add_metric, add_artifact, add_sensor_record


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "test.sqlite")
    yield conn
    conn.close()


def test_connect_creates_tables(db):
    tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"runs", "metrics", "artifacts", "sensor_data"}.issubset(tables)


def test_run_lifecycle(db):
    rid = start_run(db, name="test_run", kind="unit", params={"lr": 0.01})
    assert isinstance(rid, int)
    assert rid > 0

    add_metric(db, rid, "accuracy", 0.95, units="%", source="unit_test")
    add_artifact(db, rid, "results/out.csv", kind="csv", note="test artifact")
    finish_run(db, rid, status="finished")

    row = db.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()
    assert row[0] == "finished"


def test_metrics_stored(db):
    rid = start_run(db, "r", "t")
    add_metric(db, rid, "loss", 0.42, units="nats")
    rows = db.execute(
        "SELECT name, value, units FROM metrics WHERE run_id=?", (rid,)
    ).fetchall()
    assert rows[0] == ("loss", 0.42, "nats")


def test_artifacts_stored(db):
    rid = start_run(db, "r", "t")
    add_artifact(db, rid, "some/path.csv", kind="csv", note="n")
    rows = db.execute(
        "SELECT path, kind, note FROM artifacts WHERE run_id=?", (rid,)
    ).fetchall()
    assert rows[0] == ("some/path.csv", "csv", "n")


def test_default_status_started(db):
    rid = start_run(db, "r2", "test")
    row = db.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()
    assert row[0] == "started"


def test_multiple_runs_independent(db):
    rid1 = start_run(db, "r1", "a")
    rid2 = start_run(db, "r2", "b")
    add_metric(db, rid1, "m", 1.0)
    add_metric(db, rid2, "m", 2.0)
    v1 = db.execute(
        "SELECT value FROM metrics WHERE run_id=?", (rid1,)
    ).fetchone()[0]
    v2 = db.execute(
        "SELECT value FROM metrics WHERE run_id=?", (rid2,)
    ).fetchone()[0]
    assert v1 == 1.0 and v2 == 2.0


def test_sensor_record_stored(db):
    rid = start_run(db, "sensor_run", "integration")
    add_sensor_record(
        db,
        run_id=rid,
        timestamp=1.23,
        sensor_id="sensor-A",
        protocol="file",
        payload={"Q": 1, "Qabs": 1},
        q=1.0,
        qabs=1.0,
        source="tmp/live.jsonl",
    )
    row = db.execute(
        "SELECT sensor_id, protocol, q, qabs FROM sensor_data WHERE run_id=?",
        (rid,),
    ).fetchone()
    assert row == ("sensor-A", "file", 1.0, 1.0)
