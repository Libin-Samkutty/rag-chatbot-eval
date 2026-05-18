"""
database.py — SQLite setup and helper functions.

We use the stdlib sqlite3 module — no ORM, no extra dependencies.
The database file (eval_runs.db) is created automatically on first run.

Schema
------
conversations
    id              TEXT  PRIMARY KEY  — UUID generated per request
    question        TEXT              — the user's question
    answer          TEXT              — the model's answer
    retrieved_chunks TEXT             — JSON array of chunk texts
    chunk_sources   TEXT             — JSON array of source filenames
    faithfulness    REAL             — 0.0–1.0
    faith_reason    TEXT             — one-sentence explanation from the judge
    faith_passed    INTEGER          — 1 or 0
    relevancy       REAL
    relevancy_passed INTEGER
    precision       REAL
    precision_passed INTEGER
    latency_ms      REAL
    overall_passed  INTEGER
    created_at      TIMESTAMP        — UTC, set by SQLite DEFAULT
"""

import json
import sqlite3
from pathlib import Path


DB_PATH = Path("eval_runs.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection with row_factory set so rows act like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the conversations table if it doesn't exist yet."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id               TEXT    PRIMARY KEY,
                question         TEXT    NOT NULL,
                answer           TEXT    NOT NULL,
                retrieved_chunks TEXT    NOT NULL,
                chunk_sources    TEXT    NOT NULL,
                faithfulness     REAL,
                faith_reason     TEXT,
                faith_passed     INTEGER,
                relevancy        REAL,
                relevancy_passed INTEGER,
                precision        REAL,
                precision_passed INTEGER,
                latency_ms       REAL,
                overall_passed   INTEGER,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_conversation(row: dict) -> None:
    """Insert one completed conversation + eval result into the database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations (
                id, question, answer, retrieved_chunks, chunk_sources,
                faithfulness, faith_reason, faith_passed,
                relevancy, relevancy_passed,
                precision, precision_passed,
                latency_ms, overall_passed
            ) VALUES (
                :id, :question, :answer, :retrieved_chunks, :chunk_sources,
                :faithfulness, :faith_reason, :faith_passed,
                :relevancy, :relevancy_passed,
                :precision, :precision_passed,
                :latency_ms, :overall_passed
            )
        """, {
            **row,
            # Serialise lists to JSON strings for storage
            "retrieved_chunks": json.dumps(row["retrieved_chunks"]),
            "chunk_sources": json.dumps(row["chunk_sources"]),
        })
        conn.commit()


def get_history(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` conversations as plain dicts."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM conversations
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        # Deserialise JSON strings back to Python lists
        d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
        d["chunk_sources"] = json.loads(d["chunk_sources"])
        result.append(d)

    return result
