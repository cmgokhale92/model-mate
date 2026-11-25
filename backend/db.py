"""Tiny SQLite model registry: one row per trained model."""
import json
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

JSON_FIELDS = ("schema", "metrics", "leaderboard", "feature_importance", "dropped_columns", "decisions")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    description        TEXT,
    target             TEXT NOT NULL,
    task_type          TEXT NOT NULL,
    algorithm          TEXT NOT NULL,
    primary_metric     TEXT NOT NULL,
    primary_score      REAL,
    artifact_path      TEXT NOT NULL,
    dataset_name       TEXT,
    n_rows             INTEGER,
    n_features         INTEGER,
    schema             TEXT NOT NULL,
    metrics            TEXT,
    leaderboard        TEXT,
    feature_importance TEXT,
    dropped_columns    TEXT,
    decisions          TEXT,
    created_at         TEXT NOT NULL
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA_SQL)


def _row_to_dict(row):
    d = dict(row)
    for f in JSON_FIELDS:
        if d.get(f):
            d[f] = json.loads(d[f])
    return d


def insert_model(record: dict):
    rec = dict(record)
    for f in JSON_FIELDS:
        if f in rec:
            rec[f] = json.dumps(rec[f])
    cols = ", ".join(rec.keys())
    placeholders = ", ".join("?" for _ in rec)
    with connect() as conn:
        conn.execute(f"INSERT INTO models ({cols}) VALUES ({placeholders})", tuple(rec.values()))


def list_models():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_model(model_id: str):
    with connect() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return _row_to_dict(row) if row else None


def delete_model(model_id: str):
    with connect() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
