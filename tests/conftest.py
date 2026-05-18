"""
tests/conftest.py — Shared pytest fixtures.

The `async_client` fixture gives tests an httpx AsyncClient pointed at the
FastAPI app without needing a running server.

The `mock_openai` fixture patches the AsyncOpenAI client so tests don't
make real API calls — keeping tests fast, free, and deterministic.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from main import app


def pytest_addoption(parser):
    parser.addoption(
        "--run-evals",
        action="store_true",
        default=False,
        help="Run the golden dataset eval suite (makes real OpenAI API calls)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "eval_suite: mark test as part of the golden dataset eval suite"
    )


@pytest_asyncio.fixture
async def async_client():
    """A test HTTP client that talks directly to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def sample_chunks() -> list[str]:
    """A small set of realistic context chunks for unit tests."""
    return [
        (
            "The transformer architecture was introduced in the 2017 paper "
            "'Attention Is All You Need' by Vaswani et al. It relies entirely "
            "on self-attention mechanisms, dispensing with recurrence and convolutions."
        ),
        (
            "Self-attention allows the model to weigh the importance of different "
            "words in the input sequence when encoding each word. This enables "
            "the model to capture long-range dependencies more effectively than RNNs."
        ),
        (
            "The transformer uses multi-head attention, where attention is computed "
            "in parallel across multiple representation subspaces, then concatenated "
            "and projected to the output dimension."
        ),
    ]


@pytest.fixture
def sample_question() -> str:
    return "What is the transformer architecture?"


@pytest.fixture
def sample_answer() -> str:
    return (
        "The transformer architecture, introduced in 'Attention Is All You Need' "
        "by Vaswani et al. in 2017, relies on self-attention mechanisms rather than "
        "recurrence or convolutions. It uses multi-head attention to capture "
        "relationships between words across the full input sequence."
    )


@pytest.fixture
def off_topic_answer() -> str:
    """An answer that is unrelated to the question — should fail relevancy."""
    return (
        "Photosynthesis is the process by which plants convert sunlight into "
        "chemical energy stored in glucose. It occurs in the chloroplasts."
    )


@pytest.fixture
def hallucinated_answer() -> str:
    """An answer that adds information not in the context — should fail faithfulness."""
    return (
        "The transformer architecture was invented at Google Brain in 2015 and "
        "uses LSTM cells combined with attention mechanisms. It requires a minimum "
        "of 16 GPUs to train and was originally designed for image recognition."
    )
