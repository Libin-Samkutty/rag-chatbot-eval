"""
CI regression suite — golden dataset (100 questions, 8 eval dimensions).
All tests make real API calls. Gate with --run-evals.

    pytest tests/evals/ --run-evals -v
    pytest tests/evals/ --run-evals --domain ww1 -v
"""

import json
import pytest
import httpx
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
API_BASE = "http://localhost:8000"
HTTP_TIMEOUT = 120.0


def _load(domain=None):
    data = json.loads(GOLDEN_PATH.read_text())
    return [q for q in data if not domain or q["domain"] == domain]


def _chat(question, domain_filter=None):
    with httpx.Client(timeout=HTTP_TIMEOUT) as c:
        r = c.post(
            f"{API_BASE}/api/chat",
            json={"question": question, "domain_filter": domain_filter},
        )
        r.raise_for_status()
        return r.json()


@pytest.mark.eval_suite
def test_golden_dataset_overall_pass_rate(run_evals, domain):
    """
    Run all golden dataset questions and assert a minimum 80% overall pass rate.

    Top-level regression test: if the pass rate drops below 80%, something is
    wrong with the RAG pipeline, the knowledge base, or the eval thresholds.
    """
    questions = _load(domain=domain)
    assert questions, f"No questions loaded (domain={domain!r})"

    passed = 0
    total = len(questions)

    for q in questions:
        data = _chat(q["question"], domain_filter=q.get("domain"))
        if data["eval_result"]["overall_passed"]:
            passed += 1

    pct = passed / total * 100
    print(f"\n{passed}/{total} passed ({pct:.1f}%)")

    assert passed / total >= 0.80, (
        f"Overall pass rate {passed}/{total} ({pct:.1f}%) is below the 80% threshold."
    )


@pytest.mark.eval_suite
def test_faithfulness_regression(run_evals, domain):
    """
    Non-adversarial questions must achieve 100% faithfulness pass rate.

    Faithfulness is the most important safety property — a grounded model
    must never hallucinate on factual history questions.
    """
    questions = [
        q for q in _load(domain=domain)
        if q.get("question_type") != "adversarial"
    ]
    assert questions, "No non-adversarial questions found in dataset"

    failures = []
    for q in questions:
        data = _chat(q["question"], domain_filter=q.get("domain"))
        faith = data["eval_result"]["faithfulness"]
        if not faith["passed"]:
            failures.append({
                "id": q.get("id", "?"),
                "question": q["question"],
                "reason": faith.get("reason", ""),
            })

    if failures:
        for f in failures:
            print(f"\nFAIL [{f['id']}]: {f['question']}")
            print(f"  Reason: {f['reason']}")

    assert not failures, (
        f"Faithfulness failures on {len(failures)} non-adversarial question(s). "
        "See printed output above."
    )


@pytest.mark.eval_suite
def test_adversarial_refusal(run_evals):
    """
    Adversarial (out-of-scope) questions should produce low answer relevancy,
    confirming the model refuses to fabricate answers outside the knowledge base.
    """
    questions = [q for q in _load() if q.get("question_type") == "adversarial"]
    if not questions:
        pytest.skip("No adversarial questions in dataset")

    low_relevancy_count = 0
    for q in questions:
        data = _chat(q["question"])
        items = data["eval_result"]["answer_relevancy"].get("items", [])
        # Check for relevancy_addresses_question item with result=False
        addresses_item = next(
            (it for it in items if it.get("key") == "relevancy_addresses_question"),
            None,
        )
        if addresses_item is not None and not addresses_item.get("result", True):
            low_relevancy_count += 1

    total = len(questions)
    print(
        f"\n{low_relevancy_count}/{total} adversarial questions correctly produced "
        "low relevancy (relevancy_addresses_question=False)"
    )

    # All adversarial questions should trigger low relevancy
    assert low_relevancy_count == total, (
        f"Only {low_relevancy_count}/{total} adversarial questions triggered "
        "low relevancy. The model may be answering out-of-scope questions."
    )


@pytest.mark.eval_suite
def test_context_recall_floor(run_evals, domain):
    """
    Multi-hop and causal questions require high context recall — the retrieved
    chunks must cover all key claims needed to answer the question.
    """
    questions = [
        q for q in _load(domain=domain)
        if q.get("question_type") in ("multi_hop", "causal")
    ]
    if not questions:
        pytest.skip("No multi_hop or causal questions in dataset")

    failures = []
    for q in questions:
        data = _chat(q["question"], domain_filter=q.get("domain"))
        recall = data["eval_result"]["context_recall"]
        if not recall["passed"]:
            failures.append({
                "id": q.get("id", "?"),
                "question": q["question"],
                "reason": recall.get("reason", ""),
            })

    if failures:
        for f in failures:
            print(f"\nRECALL FAIL [{f['id']}]: {f['question']}")
            print(f"  Reason: {f['reason']}")

    assert not failures, (
        f"Context recall failures on {len(failures)} multi-hop/causal question(s). "
        "The knowledge base may need more coverage."
    )


@pytest.mark.eval_suite
def test_per_domain_pass_rate(run_evals):
    """
    Each domain must achieve at least 75% overall pass rate independently.

    This catches domain-specific regressions that would otherwise be masked
    by the aggregate pass rate.
    """
    domains = ["ww1", "ww2", "historical_figures", "revolutions"]
    domain_results = {}

    for d in domains:
        questions = _load(domain=d)
        if not questions:
            print(f"\n  {d}: no questions — skipping")
            continue

        passed = sum(
            1 for q in questions
            if _chat(q["question"], domain_filter=d)["eval_result"]["overall_passed"]
        )
        total = len(questions)
        pct = passed / total * 100
        domain_results[d] = (passed, total, pct)
        print(f"\n  {d}: {passed}/{total} ({pct:.1f}%)")

    failures = [
        d for d, (passed, total, pct) in domain_results.items()
        if passed / total < 0.75
    ]

    assert not failures, (
        f"Domain(s) below 75% pass rate: {failures}. "
        "See printed summary above."
    )
