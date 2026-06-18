"""
golden_dataset/sme_review.py — Two-agent AI SME review of the golden dataset.

Two independent AI reviewers validate each Q&A pair:
  SME1: Anthropic Claude via Vertex AI (AsyncAnthropicVertex)
  SME2: Google Gemini via Vertex AI   (google.genai.Client)

Both use the same GCP service account credentials but different model families,
preventing preference leakage (ICLR 2026 validated risk).

Disagreement handling:
  - When SME1 and SME2 disagree on any checklist item → needs_human_review = True
  - Random 10% of agreeing pairs are also flagged for spot-check

Output:
  golden_dataset/golden_dataset.json — updated in-place with sme1_verdict,
  sme2_verdict, sme1_reasoning, sme2_reasoning, needs_human_review, and
  reviewer_notes fields populated.

Usage (run from project root):
    python -m golden_dataset.sme_review
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import random
import sys
from typing import Any

from config import settings, load_vertex_credentials

logger = logging.getLogger(__name__)

_HERE = pathlib.Path(__file__).parent
_GOLDEN_PATH = _HERE.parent / "tests" / "evals" / "golden_dataset.json"

# ---------------------------------------------------------------------------
# SME prompt template (from AGENT2_QA.md)
# ---------------------------------------------------------------------------

_SME_PROMPT = """You are a World History fact-checker. Given the following question and \
reference answer, verify whether every factual claim in the answer is \
accurate and complete based on your knowledge of the topic.

Question: {question}
Reference Answer: {reference_answer}

For each of the following checklist items, respond Yes or No only:
1. Does the answer contain only verifiable historical facts?
2. Are all dates and years accurate?
3. Are all named people, places, and events correctly named?
4. Are any statistics or numbers present and correct?
5. Is the answer complete — does it address all parts of the question?

Respond in JSON: {{"item1": true/false, "item2": true/false, "item3": true/false, \
"item4": true/false, "item5": true/false, \
"reasoning": "PASS: <one sentence why all items passed> OR FAIL: <one sentence on the main failing item>"}}"""

# ---------------------------------------------------------------------------
# SME response parsing
# ---------------------------------------------------------------------------

def _parse_sme_response(raw: str) -> tuple[dict[str, bool] | None, str]:
    """
    Parse a JSON response from an SME into (items_dict, reasoning_str).

    items_dict is None if parsing fails. Handles three response shapes:
    - Pure JSON object
    - JSON wrapped in markdown code fences
    - Prose with an embedded JSON object (Claude sometimes adds preamble)
    """
    import re

    try:
        text = raw.strip()
        # Strip markdown code fences.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()
        # Try direct parse first; fall back to extracting the first {...} block.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group())
        items = {
            k: bool(v)
            for k, v in data.items()
            if k.startswith("item")
        }
        reasoning = str(data.get("reasoning", ""))
        return items, reasoning
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning("Could not parse SME response: %s | raw: %.200s", exc, raw)
        return None, ""


def _verdict_from_items(items: dict[str, bool] | None) -> str:
    """Return 'pass', 'fail', or 'error' based on item verdicts."""
    if items is None:
        return "error"
    return "pass" if all(items.values()) else "fail"


def _items_agree(items1: dict[str, bool] | None, items2: dict[str, bool] | None) -> bool:
    """Return True if both SMEs agree on every item."""
    if items1 is None or items2 is None:
        return False
    # Both must have the same keys and same values.
    if set(items1.keys()) != set(items2.keys()):
        return False
    return all(items1[k] == items2[k] for k in items1)


# ---------------------------------------------------------------------------
# SME1: Claude via AsyncAnthropicVertex
# ---------------------------------------------------------------------------

async def _sme1_review(
    question: str,
    reference_answer: str,
    client: Any,
) -> tuple[dict[str, bool] | None, str]:
    """
    Run the SME1 (Claude) review for one Q&A pair.

    Returns (items_dict, reasoning_str); items_dict is None on failure.
    """
    prompt = _SME_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
    )
    try:
        response = await client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            system="You are a JSON API. Respond only with a valid JSON object. No preamble, explanation, or markdown.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text if response.content else ""
        return _parse_sme_response(raw)
    except Exception as exc:
        logger.warning("SME1 (Claude) failed: %s", exc)
        return None, ""


# ---------------------------------------------------------------------------
# SME2: Gemini via google.genai
# ---------------------------------------------------------------------------

async def _sme2_review(
    question: str,
    reference_answer: str,
    client: Any,
) -> tuple[dict[str, bool] | None, str]:
    """
    Run the SME2 (Gemini) review for one Q&A pair.

    google.genai is synchronous — we wrap with asyncio.to_thread.
    Returns (items_dict, reasoning_str); items_dict is None on failure.
    """
    prompt = _SME_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
    )

    def _call_gemini() -> str:
        from google.genai import types as genai_types

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        return response.text or ""

    try:
        raw = await asyncio.to_thread(_call_gemini)
        return _parse_sme_response(raw)
    except Exception as exc:
        logger.warning("SME2 (Gemini) failed: %s", exc)
        return None, ""


# ---------------------------------------------------------------------------
# Per-entry review
# ---------------------------------------------------------------------------

async def _review_entry(
    entry: dict[str, Any],
    sme1_client: Any,
    sme2_client: Any,
    spot_check_ids: set[str],
) -> dict[str, Any]:
    """
    Run both SMEs on one golden dataset entry and update it in-place.
    """
    question: str = entry.get("question", "")
    reference: str = entry.get("reference_answer", "")
    entry_id: str = entry.get("id", "unknown")

    (sme1_items, sme1_reasoning), (sme2_items, sme2_reasoning) = await asyncio.gather(
        _sme1_review(question, reference, sme1_client),
        _sme2_review(question, reference, sme2_client),
    )

    sme1_verdict = _verdict_from_items(sme1_items)
    sme2_verdict = _verdict_from_items(sme2_items)

    agree = _items_agree(sme1_items, sme2_items)
    needs_human = not agree or entry_id in spot_check_ids

    reviewer_notes = ""
    if not agree:
        # Describe which items differ.
        if sme1_items and sme2_items:
            differing = [
                k for k in sme1_items
                if sme1_items.get(k) != sme2_items.get(k)
            ]
            reviewer_notes = f"SME disagreement on: {', '.join(differing)}"
        else:
            reviewer_notes = "SME disagreement: one or both SMEs returned an error"
    elif entry_id in spot_check_ids:
        reviewer_notes = "Flagged for random spot-check (10% sample)."

    entry["sme1_verdict"] = sme1_verdict
    entry["sme2_verdict"] = sme2_verdict
    entry["sme1_reasoning"] = sme1_reasoning
    entry["sme2_reasoning"] = sme2_reasoning
    entry["needs_human_review"] = needs_human
    entry["reviewer_notes"] = reviewer_notes

    logger.debug(
        "%s: SME1=%s SME2=%s agree=%s human=%s",
        entry_id, sme1_verdict, sme2_verdict, agree, needs_human,
    )
    return entry


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_sme_review() -> None:
    """
    Load golden_dataset.json, run both SMEs on every entry, and write back.
    """
    if not _GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {_GOLDEN_PATH}. "
            "Run golden_dataset/generator.py first."
        )

    payload = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = payload if isinstance(payload, list) else payload.get("entries", [])

    if not entries:
        logger.error("Golden dataset has no entries.")
        sys.exit(1)

    logger.info("Starting SME review for %d entries.", len(entries))

    # ------------------------------------------------------------------ #
    # Credentials & clients
    # ------------------------------------------------------------------ #
    credentials = load_vertex_credentials()

    from anthropic import AsyncAnthropicVertex
    from google import genai

    sme1_client = AsyncAnthropicVertex(
        project_id=settings.vertex_project_id,
        region=settings.vertex_region,
        credentials=credentials,
    )

    sme2_client = genai.Client(
        vertexai=True,
        project=settings.vertex_project_id,
        location=settings.vertex_region,
        credentials=credentials,
    )

    # ------------------------------------------------------------------ #
    # 10% random spot-check selection
    # ------------------------------------------------------------------ #
    all_ids = [e.get("id", str(i)) for i, e in enumerate(entries)]
    n_spot = max(1, round(len(entries) * 0.10))
    spot_check_ids: set[str] = set(random.sample(all_ids, n_spot))
    logger.info("Flagging %d entries for random spot-check.", n_spot)

    # ------------------------------------------------------------------ #
    # Run reviews concurrently (batch to avoid overwhelming the APIs)
    # ------------------------------------------------------------------ #
    BATCH_SIZE = 5  # Concurrent API calls per batch

    updated_entries: list[dict[str, Any]] = []
    for batch_start in range(0, len(entries), BATCH_SIZE):
        batch = entries[batch_start: batch_start + BATCH_SIZE]
        results = await asyncio.gather(
            *[
                _review_entry(entry, sme1_client, sme2_client, spot_check_ids)
                for entry in batch
            ]
        )
        updated_entries.extend(results)
        logger.info(
            "Reviewed %d/%d entries...",
            min(batch_start + BATCH_SIZE, len(entries)),
            len(entries),
        )

    # ------------------------------------------------------------------ #
    # Compute disagreement statistics
    # ------------------------------------------------------------------ #
    disagreements = sum(
        1
        for e in updated_entries
        if e.get("needs_human_review")
        and "disagreement" in e.get("reviewer_notes", "").lower()
    )
    print(f"\nSME review complete.")
    print(f"  Total entries reviewed: {len(updated_entries)}")
    print(f"  Disagreements (needs human review): {disagreements}")
    print(f"  Random spot-checks flagged: {n_spot}")
    print(f"  Total needs_human_review: {sum(1 for e in updated_entries if e.get('needs_human_review'))}")

    # ------------------------------------------------------------------ #
    # Write back
    # ------------------------------------------------------------------ #
    output = updated_entries if isinstance(payload, list) else {**payload, "entries": updated_entries}
    _GOLDEN_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Updated golden dataset written to %s.", _GOLDEN_PATH)
    print(f"\nUpdated dataset saved to {_GOLDEN_PATH}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_sme_review())
    except Exception as exc:
        logger.error("SME review failed: %s", exc, exc_info=True)
        sys.exit(1)
