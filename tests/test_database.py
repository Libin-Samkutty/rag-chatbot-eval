"""Unit tests for database.py using a temp file (not :memory: to support ALTER TABLE)."""

import json
import pytest
from pathlib import Path


ROW = {
    "id": "test-uuid",
    "question": "When did WW1 start?",
    "answer": "In 1914.",
    "retrieved_chunks": ["chunk text"],
    "chunk_sources": ["ww1_world_war_i.txt"],
    "faithfulness": 0.9,
    "faith_reason": "Grounded in context.",
    "faith_passed": 1,
    "relevancy": 0.8,
    "relevancy_passed": 1,
    "precision": 0.7,
    "precision_passed": 1,
    "latency_ms": 1500.0,
    "overall_passed": 1,
    "domain_tag": "ww1",
    "answer_relevancy_passed": 1,
    "completeness_passed": 1,
    "context_recall_passed": 1,
    "coherence_passed": 1,
    "historical_balance_passed": 1,
    "toxicity_passed": 1,
    "checklist_json": {"faithfulness": {"passed": True}},
}


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    import database
    monkeypatch.setattr(database, "DB_PATH", db_file)
    database.init_db()
    yield db_file


# --- Schema tests ---

class TestInitDb:

    def test_conversations_table_exists(self, tmp_db):
        """init_db() must create the conversations table."""
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "conversations" in tables

    def test_all_expected_columns_present(self, tmp_db):
        """Conversations table must contain every expected column."""
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        conn.close()

        expected = {
            "id", "question", "answer", "retrieved_chunks", "chunk_sources",
            "faithfulness", "faith_reason", "faith_passed",
            "relevancy", "relevancy_passed",
            "precision", "precision_passed",
            "latency_ms", "overall_passed", "created_at",
            # New 8-dimension columns
            "domain_tag", "answer_relevancy_passed", "completeness_passed",
            "context_recall_passed", "coherence_passed",
            "historical_balance_passed", "toxicity_passed",
            "checklist_json",
        }
        assert expected.issubset(columns), (
            f"Missing columns: {expected - columns}"
        )

    def test_init_db_idempotent(self, tmp_db):
        """Calling init_db() twice must not raise an error."""
        import database
        database.init_db()  # second call — CREATE TABLE IF NOT EXISTS should no-op


# --- save_conversation tests ---

class TestSaveConversation:

    def test_save_inserts_row(self, tmp_db):
        """save_conversation() must insert exactly one row."""
        import database, sqlite3
        database.save_conversation(ROW)
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        conn.close()
        assert count == 1

    def test_save_stores_all_fields(self, tmp_db):
        """Saved row must contain the correct values for key fields."""
        import database, sqlite3
        database.save_conversation(ROW)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (ROW["id"],)
        ).fetchone()
        conn.close()

        assert row is not None
        cols = [description[0] for description in conn.description] if False else None
        # Use a fresh query with column names via sqlite3.Row
        conn2 = sqlite3.connect(tmp_db)
        conn2.row_factory = sqlite3.Row
        r = conn2.execute(
            "SELECT * FROM conversations WHERE id = ?", (ROW["id"],)
        ).fetchone()
        conn2.close()

        assert r["question"] == ROW["question"]
        assert r["answer"] == ROW["answer"]
        assert r["domain_tag"] == ROW["domain_tag"]
        assert r["faith_passed"] == ROW["faith_passed"]
        assert r["overall_passed"] == ROW["overall_passed"]
        assert r["latency_ms"] == ROW["latency_ms"]

    def test_save_serialises_checklist_json(self, tmp_db):
        """checklist_json should be stored as a JSON string."""
        import database, sqlite3
        database.save_conversation(ROW)
        conn = sqlite3.connect(tmp_db)
        raw = conn.execute(
            "SELECT checklist_json FROM conversations WHERE id = ?", (ROW["id"],)
        ).fetchone()[0]
        conn.close()

        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "faithfulness" in parsed

    def test_save_serialises_retrieved_chunks(self, tmp_db):
        """retrieved_chunks should be stored as a JSON array string."""
        import database, sqlite3
        database.save_conversation(ROW)
        conn = sqlite3.connect(tmp_db)
        raw = conn.execute(
            "SELECT retrieved_chunks FROM conversations WHERE id = ?", (ROW["id"],)
        ).fetchone()[0]
        conn.close()

        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert parsed == ROW["retrieved_chunks"]


# --- get_history tests ---

class TestGetHistory:

    def test_get_history_returns_list(self, tmp_db):
        """get_history() must return a list."""
        import database
        result = database.get_history()
        assert isinstance(result, list)

    def test_get_history_returns_rows_in_desc_order(self, tmp_db):
        """get_history() must return rows ordered most-recent first."""
        import database, time
        # Insert two rows with different IDs
        row_a = {**ROW, "id": "uuid-a", "question": "First question"}
        row_b = {**ROW, "id": "uuid-b", "question": "Second question"}
        database.save_conversation(row_a)
        database.save_conversation(row_b)

        history = database.get_history()
        assert len(history) == 2
        # Most recent (uuid-b) should be first
        assert history[0]["id"] == "uuid-b"
        assert history[1]["id"] == "uuid-a"

    def test_get_history_deserialises_retrieved_chunks(self, tmp_db):
        """retrieved_chunks must be returned as a Python list, not a JSON string."""
        import database
        database.save_conversation(ROW)
        history = database.get_history()
        assert isinstance(history[0]["retrieved_chunks"], list)

    def test_get_history_deserialises_chunk_sources(self, tmp_db):
        """chunk_sources must be returned as a Python list, not a JSON string."""
        import database
        database.save_conversation(ROW)
        history = database.get_history()
        assert isinstance(history[0]["chunk_sources"], list)


# --- get_eval_runs tests ---

class TestGetEvalRuns:

    def _insert_rows(self):
        import database
        # Insert 4 rows: 2 ww1, 1 ww2, 1 historical_figures
        for i, domain in enumerate(["ww1", "ww1", "ww2", "historical_figures"]):
            database.save_conversation({**ROW, "id": f"uuid-{i}", "domain_tag": domain})

    def test_get_eval_runs_no_filter(self, tmp_db):
        """Without domain filter, all rows are returned."""
        self._insert_rows()
        import database
        runs = database.get_eval_runs()
        assert len(runs) == 4

    def test_get_eval_runs_domain_filter_ww1(self, tmp_db):
        """domain='ww1' must return only ww1 rows."""
        self._insert_rows()
        import database
        runs = database.get_eval_runs(domain="ww1")
        assert len(runs) == 2
        assert all(r["domain_tag"] == "ww1" for r in runs)

    def test_get_eval_runs_limit(self, tmp_db):
        """limit=2 must return at most 2 rows."""
        self._insert_rows()
        import database
        runs = database.get_eval_runs(limit=2, offset=0)
        assert len(runs) == 2

    def test_get_eval_runs_offset(self, tmp_db):
        """offset=2 must skip the first 2 rows."""
        self._insert_rows()
        import database
        page1 = database.get_eval_runs(limit=2, offset=0)
        page2 = database.get_eval_runs(limit=2, offset=2)
        # No ID overlap between pages
        ids_p1 = {r["id"] for r in page1}
        ids_p2 = {r["id"] for r in page2}
        assert ids_p1.isdisjoint(ids_p2)
        assert len(page2) == 2
