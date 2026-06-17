"""
tests/conftest.py — Shared pytest fixtures.

The `async_client` fixture gives tests an httpx AsyncClient pointed at the
FastAPI app without needing a running server.

The `run_evals` fixture gates the golden dataset suite behind --run-evals.
The `domain` fixture exposes the --domain filter value to eval tests.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import app


def pytest_addoption(parser):
    parser.addoption(
        "--run-evals",
        action="store_true",
        default=False,
        help="Run the golden dataset eval suite (makes real API calls)",
    )
    parser.addoption(
        "--domain",
        action="store",
        default=None,
        help="Filter golden dataset eval suite to a single domain (e.g. ww1)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "eval_suite: mark test as part of the golden dataset eval suite"
    )
    config.addinivalue_line(
        "markers", "eval: mark test as an evaluation test requiring --run-evals"
    )


@pytest.fixture
def run_evals(request):
    """Skip this test unless --run-evals was passed on the command line."""
    if not request.config.getoption("--run-evals"):
        pytest.skip("Pass --run-evals to run the golden dataset suite")


@pytest.fixture
def domain(request) -> str | None:
    """Return the --domain option value (None means all domains)."""
    return request.config.getoption("--domain")


@pytest_asyncio.fixture
async def async_client():
    """A test HTTP client that talks directly to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def sample_chunks() -> list[dict]:
    """A small set of realistic context chunks for unit tests."""
    return [
        {
            "text": (
                "World War I began in 1914 following the assassination of Archduke "
                "Franz Ferdinand of Austria. The conflict drew in the major European "
                "powers and eventually became a global war lasting until 1918."
            ),
            "source": "ww1_world_war_i.txt",
            "domain_tag": "ww1",
        },
        {
            "text": (
                "The assassination of Archduke Franz Ferdinand on 28 June 1914 in "
                "Sarajevo, Bosnia, was carried out by Gavrilo Princip, a Bosnian Serb "
                "nationalist. This event triggered the July Crisis and ultimately the "
                "outbreak of the First World War."
            ),
            "source": "ww1_assassination_franz_ferdinand.txt",
            "domain_tag": "ww1",
        },
    ]


@pytest.fixture
def sample_question() -> str:
    return "What caused the outbreak of World War I?"


@pytest.fixture
def sample_answer() -> str:
    return (
        "World War I was triggered by the assassination of Archduke Franz Ferdinand "
        "of Austria in Sarajevo on 28 June 1914. The killing, carried out by Bosnian "
        "Serb nationalist Gavrilo Princip, set off the July Crisis, drawing the major "
        "European powers into a conflict that lasted until 1918."
    )


@pytest.fixture
def off_topic_answer() -> str:
    """An answer that is unrelated to the question — should fail relevancy."""
    return (
        "The Battle of Waterloo was fought on 18 June 1815 near Waterloo in present-day "
        "Belgium. Napoleon Bonaparte was decisively defeated by the Duke of Wellington "
        "and the Prussian army under Field Marshal Blücher."
    )


@pytest.fixture
def hallucinated_answer() -> str:
    """An answer that adds information not in the context — should fail faithfulness."""
    return (
        "World War I began in 1912 when the United States declared war on Germany "
        "following a naval blockade of New York Harbor. The conflict was resolved "
        "within six months by the Treaty of Paris, signed by 47 nations."
    )
