"""
tests/evals/test_eval_suite.py — Golden dataset evaluation suite.

This is the "CI eval" demonstration. It:
  1. Loads 10 pre-written questions from golden_dataset.json
  2. Sends each through the full /api/chat endpoint (RAG + generation + eval)
  3. Asserts that the eval results match the expected pass/fail for each case

This is the pattern you would use in a real CI pipeline:
  - A dataset of known questions with expected eval outcomes
  - Automated assertions that catch regressions
  - Run on every PR to catch eval metric drift

WARNING: This test suite makes real OpenAI API calls. Each run costs ~$0.01.
Add --run-evals to the pytest command to opt in:
  pytest tests/evals/ --run-evals -v

Without the flag, all tests in this file are skipped.
"""

import json
from pathlib import Path

import pytest
import pytest_asyncio

# --- Load golden dataset ---

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


@pytest.fixture(scope="module")
def golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


# --- Tests ---

@pytest.mark.eval_suite
@pytest.mark.asyncio
async def test_golden_dataset_pass_rate(async_client, golden_dataset, request):
    """
    Run all golden dataset questions and assert a minimum overall pass rate.

    This is the top-level regression test: if the pass rate drops below 70%,
    something is wrong with the RAG pipeline, the knowledge base, or the
    eval thresholds.
    """
    if not request.config.getoption("--run-evals"):
        pytest.skip("Pass --run-evals to run the golden dataset suite")

    results = []

    for case in golden_dataset:
        response = await async_client.post(
            "/api/chat", json={"question": case["question"]}
        )
        assert response.status_code == 200, (
            f"Request failed for '{case['id']}': {response.text}"
        )

        data = response.json()
        eval_result = data["eval_result"]

        actual = {
            "faithfulness_pass": eval_result["faithfulness"]["passed"],
            "relevancy_pass": eval_result["answer_relevancy"]["passed"],
            "context_precision_pass": eval_result["context_precision"]["passed"],
        }
        expected = case["expected"]

        # Check each metric matches the expected outcome
        matches = all(actual[k] == expected[k] for k in expected)
        results.append({
            "id": case["id"],
            "question": case["question"],
            "actual": actual,
            "expected": expected,
            "matches": matches,
        })

        if not matches:
            print(f"\n⚠️  MISMATCH [{case['id']}]: {case['question']}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")

    pass_count = sum(1 for r in results if r["matches"])
    total = len(results)
    pass_rate = pass_count / total

    print(f"\n📊 Golden dataset pass rate: {pass_count}/{total} ({pass_rate:.0%})")

    # Assert at least 70% of cases match expectations
    assert pass_rate >= 0.70, (
        f"Golden dataset pass rate {pass_rate:.0%} is below the 70% threshold. "
        f"Check the failing cases above."
    )


@pytest.mark.eval_suite
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,expected_faith", [
    ("gd_001", True),
    ("gd_002", True),
    ("gd_007", True),   # Out-of-domain: model should refuse faithfully
    ("gd_010", True),   # Out-of-domain: model should refuse faithfully
])
async def test_faithfulness_individual_cases(
    async_client, golden_dataset, request, case_id, expected_faith
):
    """
    Test faithfulness individually for cases where we have strong expectations.
    These are the cases most important to catch regressions on.
    """
    if not request.config.getoption("--run-evals"):
        pytest.skip("Pass --run-evals to run the golden dataset suite")

    case = next(c for c in golden_dataset if c["id"] == case_id)

    response = await async_client.post(
        "/api/chat", json={"question": case["question"]}
    )
    assert response.status_code == 200

    data = response.json()
    actual_faith_pass = data["eval_result"]["faithfulness"]["passed"]

    assert actual_faith_pass == expected_faith, (
        f"Faithfulness for '{case_id}' expected {expected_faith}, "
        f"got {actual_faith_pass}. "
        f"Reason: {data['eval_result']['faithfulness']['reason']}"
    )
