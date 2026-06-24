"""Unit tests for POST /api/chat — all external calls mocked."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from eval.models import EvalResult, DimensionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dim(name: str) -> DimensionResult:
    return DimensionResult(
        name=name,
        passed=True,
        items=[],
        score=None,
        reason="ok",
        tier1_failed=[],
        tier2_pass_rate=1.0,
    )


FAKE_EVAL = EvalResult(
    faithfulness=_make_dim("faithfulness"),
    answer_relevancy=_make_dim("answer_relevancy"),
    completeness=_make_dim("completeness"),
    context_precision=_make_dim("context_precision"),
    context_recall=_make_dim("context_recall"),
    coherence=_make_dim("coherence"),
    historical_balance=_make_dim("historical_balance"),
    toxicity=_make_dim("toxicity"),
    latency_ms=500.0,
    overall_passed=True,
)

FAKE_CHUNKS = [
    {"text": "World War I began in 1914.", "source": "ww1_world_war_i.txt", "domain_tag": "ww1"},
    {"text": "The assassination of Franz Ferdinand triggered the war.", "source": "ww1_assassination.txt", "domain_tag": "ww1"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retrieve():
    with patch("routers.chat.retrieve_chunks", new_callable=AsyncMock) as m:
        m.return_value = FAKE_CHUNKS
        yield m


@pytest.fixture
def mock_run_evals():
    with patch("routers.chat.run_evals", new_callable=AsyncMock) as m:
        m.return_value = FAKE_EVAL
        yield m


@pytest.fixture
def mock_gemini():
    with patch("routers.chat._generate_answer", new_callable=AsyncMock) as m:
        m.return_value = "Test answer about WW1."
        yield m


@pytest.fixture
def mock_save():
    with patch("database.save_conversation") as m:
        yield m


@pytest.fixture
def client(mock_retrieve, mock_run_evals, mock_gemini, mock_save):
    """TestClient with all external dependencies mocked."""
    from main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatEndpoint:

    def test_basic_chat_returns_200(self, client):
        """POST /api/chat with a valid question must return 200."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        assert resp.status_code == 200

    def test_basic_chat_response_shape(self, client):
        """Response body must contain all expected top-level fields."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        data = resp.json()
        for field in ("message_id", "question", "answer", "chunks", "eval_result", "latency_ms"):
            assert field in data, f"Missing field: {field}"

    def test_chat_with_domain_filter(self, client):
        """POST /api/chat with domain_filter must still return 200."""
        resp = client.post(
            "/api/chat",
            json={"question": "What caused WW1?", "domain_filter": "ww1"},
        )
        assert resp.status_code == 200

    def test_chat_question_echoed_in_response(self, client):
        """The response must echo back the original question."""
        question = "What caused WW1?"
        resp = client.post("/api/chat", json={"question": question})
        assert resp.json()["question"] == question

    def test_chat_answer_in_response(self, client):
        """The response must include the generated answer text."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        assert resp.json()["answer"] == "Test answer about WW1."

    def test_chat_chunks_in_response(self, client):
        """The response must include the retrieved chunks list."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        chunks = resp.json()["chunks"]
        assert isinstance(chunks, list)
        assert len(chunks) == 2

    def test_chat_eval_result_has_all_dimensions(self, client):
        """eval_result must contain all 8 evaluation dimensions."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        eval_result = resp.json()["eval_result"]
        for dim in (
            "faithfulness", "answer_relevancy", "completeness",
            "context_precision", "context_recall", "coherence",
            "historical_balance", "toxicity",
        ):
            assert dim in eval_result, f"Missing eval dimension: {dim}"

    def test_chat_eval_overall_passed(self, client):
        """overall_passed must be True when all dimensions pass."""
        resp = client.post("/api/chat", json={"question": "What caused WW1?"})
        assert resp.json()["eval_result"]["overall_passed"] is True

    def test_empty_question_returns_422(self, client):
        """POST /api/chat with a non-string question must return 422 (validation error)."""
        # Pass a non-string type for question — this will fail Pydantic validation
        resp = client.post("/api/chat", json={"question": None})
        assert resp.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        """POST /api/chat with no question field must return 422."""
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """GET /api/health must return 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self, client):
        """GET /api/health body must have status == 'ok'."""
        resp = client.get("/api/health")
        assert resp.json()["status"] == "ok"


class TestHistoryEndpoint:

    def test_history_returns_200(self, client):
        """GET /api/history must return 200."""
        with patch("database.get_history", return_value=[]):
            resp = client.get("/api/history")
        assert resp.status_code == 200

    def test_history_returns_list(self, client):
        """GET /api/history must return a list."""
        with patch("database.get_history", return_value=[]):
            resp = client.get("/api/history")
        assert isinstance(resp.json(), list)


class TestEvalRunsEndpoint:

    def test_eval_runs_returns_200(self, client):
        """GET /api/eval-runs must return 200."""
        with patch("database.get_eval_runs", return_value=[]):
            resp = client.get("/api/eval-runs")
        assert resp.status_code == 200

    def test_eval_runs_returns_dict_with_runs(self, client):
        """GET /api/eval-runs must return a dict with a 'runs' key."""
        with patch("database.get_eval_runs", return_value=[]):
            resp = client.get("/api/eval-runs")
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)
