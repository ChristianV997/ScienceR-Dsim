from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT, kind TEXT, started_at REAL, finished_at REAL,
  status TEXT, params_json TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, name TEXT, value REAL, units TEXT, source TEXT, extra_json TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, path TEXT, kind TEXT, note TEXT
);
"""

def connect(db_path: str | Path):
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn

def start_run(conn, name: str, kind: str, params: dict | None = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs(name, kind, started_at, status, params_json) VALUES(?,?,?,?,?)",
        (name, kind, time.time(), "started", json.dumps(params or {})),
    )
    conn.commit()
    return int(cur.lastrowid)

def finish_run(conn, run_id: int, status: str = "finished"):
    conn.execute("UPDATE runs SET finished_at=?, status=? WHERE id=?", (time.time(), status, run_id))
    conn.commit()

def add_metric(conn, run_id: int, name: str, value: float, units: str = "", source: str = "", extra: dict | None = None):
    conn.execute(
        "INSERT INTO metrics(run_id,name,value,units,source,extra_json) VALUES(?,?,?,?,?,?)",
        (run_id, name, float(value), units, source, json.dumps(extra or {})),
    )
    conn.commit()

def add_artifact(conn, run_id: int, path: str | Path, kind: str = "", note: str = ""):
    conn.execute(
        "INSERT INTO artifacts(run_id,path,kind,note) VALUES(?,?,?,?)",
        (run_id, str(path), kind, note),
    )
    conn.commit()
