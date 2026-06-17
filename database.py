"""
database.py — SQLite setup and helper functions.

Schema (conversations table):
  Original columns (kept for backward compatibility):
    id, question, answer, retrieved_chunks, chunk_sources,
    faithfulness, faith_reason, faith_passed,
    relevancy, relevancy_passed, precision, precision_passed,
    latency_ms, overall_passed, created_at

  Added 2026-06-17 — 8-dimension eval overhaul:
    domain_tag              TEXT    — ww1 | ww2 | historical_figures | revolutions | general
    answer_relevancy_passed INTEGER — pass/fail for answer_relevancy dimension
    completeness_passed     INTEGER
    context_recall_passed   INTEGER
    coherence_passed        INTEGER
    historical_balance_passed INTEGER
    toxicity_passed         INTEGER
    checklist_json          TEXT    — full JSON blob of all DimensionResult data
"""

import json
import sqlite3
from pathlib import Path


DB_PATH = Path("eval_runs.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the conversations table and add any missing columns."""
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
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Added 2026-06-17
                domain_tag              TEXT,
                answer_relevancy_passed INTEGER,
                completeness_passed     INTEGER,
                context_recall_passed   INTEGER,
                coherence_passed        INTEGER,
                historical_balance_passed INTEGER,
                toxicity_passed         INTEGER,
                checklist_json          TEXT
            )
        """)
        # Idempotent migrations for existing databases
        _add_column_if_missing(conn, "domain_tag", "TEXT")
        _add_column_if_missing(conn, "answer_relevancy_passed", "INTEGER")
        _add_column_if_missing(conn, "completeness_passed", "INTEGER")
        _add_column_if_missing(conn, "context_recall_passed", "INTEGER")
        _add_column_if_missing(conn, "coherence_passed", "INTEGER")
        _add_column_if_missing(conn, "historical_balance_passed", "INTEGER")
        _add_column_if_missing(conn, "toxicity_passed", "INTEGER")
        _add_column_if_missing(conn, "checklist_json", "TEXT")
        conn.commit()


def _add_column_if_missing(conn: sqlite3.Connection, column: str, col_type: str) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} {col_type}")


def save_conversation(row: dict) -> None:
    """Insert one completed conversation + eval result into the database."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations (
                id, question, answer, retrieved_chunks, chunk_sources,
                faithfulness, faith_reason, faith_passed,
                relevancy, relevancy_passed,
                precision, precision_passed,
                latency_ms, overall_passed,
                domain_tag, answer_relevancy_passed, completeness_passed,
                context_recall_passed, coherence_passed,
                historical_balance_passed, toxicity_passed,
                checklist_json
            ) VALUES (
                :id, :question, :answer, :retrieved_chunks, :chunk_sources,
                :faithfulness, :faith_reason, :faith_passed,
                :relevancy, :relevancy_passed,
                :precision, :precision_passed,
                :latency_ms, :overall_passed,
                :domain_tag, :answer_relevancy_passed, :completeness_passed,
                :context_recall_passed, :coherence_passed,
                :historical_balance_passed, :toxicity_passed,
                :checklist_json
            )
        """, {
            **row,
            "retrieved_chunks": json.dumps(row.get("retrieved_chunks", [])),
            "chunk_sources": json.dumps(row.get("chunk_sources", [])),
            "checklist_json": json.dumps(row.get("checklist_json")) if row.get("checklist_json") is not None else None,
            # Defaults for new columns — callers should supply these
            "domain_tag": row.get("domain_tag"),
            "answer_relevancy_passed": row.get("answer_relevancy_passed"),
            "completeness_passed": row.get("completeness_passed"),
            "context_recall_passed": row.get("context_recall_passed"),
            "coherence_passed": row.get("coherence_passed"),
            "historical_balance_passed": row.get("historical_balance_passed"),
            "toxicity_passed": row.get("toxicity_passed"),
        })
        conn.commit()


def get_history(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` conversations as plain dicts."""
    limit = min(limit, 100)
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM conversations
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
        """, (limit,)).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
        d["chunk_sources"] = json.loads(d["chunk_sources"])
        if d.get("checklist_json"):
            d["checklist_json"] = json.loads(d["checklist_json"])
        result.append(d)
    return result


def get_eval_runs(
    domain: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return paginated eval run records, optionally filtered by domain_tag.

    Args:
        domain: If set, filter to rows with this domain_tag.
        limit:  Max records to return (capped at 200).
        offset: Skip this many records (for pagination).
    """
    limit = min(limit, 200)
    with get_connection() as conn:
        if domain:
            rows = conn.execute("""
                SELECT id, question, domain_tag, latency_ms, overall_passed,
                       faith_passed, answer_relevancy_passed, completeness_passed,
                       context_recall_passed, coherence_passed,
                       historical_balance_passed, toxicity_passed,
                       created_at
                FROM conversations
                WHERE domain_tag = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (domain, limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, question, domain_tag, latency_ms, overall_passed,
                       faith_passed, answer_relevancy_passed, completeness_passed,
                       context_recall_passed, coherence_passed,
                       historical_balance_passed, toxicity_passed,
                       created_at
                FROM conversations
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()

    return [dict(row) for row in rows]
