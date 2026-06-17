"""
golden_dataset/generator.py — RAGAS TestsetGenerator pipeline.

Generates 100 Q&A pairs from the knowledge base articles using Claude via
Vertex AI for question/answer generation (different family from the judge).

Usage:
    python -m golden_dataset.generator
    python -m golden_dataset.generator --use-saved-kg   # reload saved knowledge graph

Output files:
    golden_dataset/knowledge_graph.json   — reusable knowledge graph (always written)
    golden_dataset/raw_generated.json     — raw RAGAS output (NEVER overwritten)
    golden_dataset/golden_dataset.json    — working copy (overwritten on each run)

Question type mapping from RAGAS synthesizer types:
    simple       → direct_factual
    reasoning    → causal
    multi_context→ multi_hop
    conditional  → comparative
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime
from typing import Any

from config import settings, load_vertex_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
_KNOWLEDGE_DIR = pathlib.Path(settings.knowledge_path)
_KG_PATH = _HERE / "knowledge_graph.json"
_RAW_PATH = _HERE / "raw_generated.json"
_GOLDEN_PATH = _HERE / "golden_dataset.json"

# ---------------------------------------------------------------------------
# RAGAS question type → our question type
# ---------------------------------------------------------------------------
_TYPE_MAP: dict[str, str] = {
    "simple": "direct_factual",
    "reasoning": "causal",
    "multi_context": "multi_hop",
    "conditional": "comparative",
    # Fallback for any other synthesizer type
    "abstract": "direct_factual",
    "specific": "direct_factual",
    "common_topic": "direct_factual",
}


def _domain_tag_from_filename(filename: str) -> str:
    """Extract domain tag from 'ww1_world_war_i.txt' → 'ww1'."""
    stem = pathlib.Path(filename).stem
    parts = stem.split("_", 1)
    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------------
# Knowledge base loader
# ---------------------------------------------------------------------------

def load_documents() -> list[dict[str, Any]]:
    """
    Load all .txt files from knowledge/ as simple dicts with page_content
    and metadata (source, domain_tag).

    Returns a list of document dicts compatible with RAGAS KnowledgeGraph.
    """
    docs = []
    txt_files = sorted(_KNOWLEDGE_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {_KNOWLEDGE_DIR}. "
            "Run: python scripts/fetch_kb.py --refresh"
        )

    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        docs.append({
            "page_content": text,
            "metadata": {
                "source": path.name,
                "domain_tag": _domain_tag_from_filename(path.name),
            },
        })

    logger.info("Loaded %d knowledge base documents.", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph(docs: list[dict[str, Any]], claude_llm: Any, embeddings: Any) -> Any:
    """
    Build and enrich a RAGAS KnowledgeGraph from the loaded documents.

    Applies default_transforms (entity extraction, summarisation, themes)
    using Claude as the LLM.
    """
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms, default_transforms

    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc["page_content"],
                    "document_metadata": doc["metadata"],
                },
            )
        )

    logger.info("Applying knowledge graph transforms (this may take several minutes)...")
    transforms = default_transforms(llm=claude_llm, embedding_model=embeddings)
    apply_transforms(kg, transforms)

    kg.save(str(_KG_PATH))
    logger.info("Knowledge graph saved to %s.", _KG_PATH)
    return kg


def load_knowledge_graph() -> Any:
    """Load a previously-saved knowledge graph from disk."""
    from ragas.testset.graph import KnowledgeGraph

    if not _KG_PATH.exists():
        raise FileNotFoundError(
            f"No saved knowledge graph at {_KG_PATH}. "
            "Run without --use-saved-kg first."
        )
    logger.info("Loading saved knowledge graph from %s.", _KG_PATH)
    return KnowledgeGraph.load(str(_KG_PATH))


# ---------------------------------------------------------------------------
# Dataset entry formatting
# ---------------------------------------------------------------------------

def _format_entry(idx: int, row: Any) -> dict[str, Any]:
    """
    Convert a RAGAS TestsetSample row into our golden dataset entry schema.

    Adversarial and ambiguous question types are not generated automatically
    by RAGAS — those must be added manually after generation.
    """
    ragas_type = getattr(row, "synthesizer_name", "simple") or "simple"
    # Normalise: RAGAS may return e.g. "SingleHopSpecificQuerySynthesizer"
    ragas_type_lower = ragas_type.lower()
    if "reasoning" in ragas_type_lower or "multi_hop" in ragas_type_lower:
        our_type = "causal"
    elif "multi_context" in ragas_type_lower or "multihop" in ragas_type_lower:
        our_type = "multi_hop"
    elif "conditional" in ragas_type_lower or "comparative" in ragas_type_lower:
        our_type = "comparative"
    else:
        our_type = "direct_factual"

    question: str = getattr(row, "user_input", "") or ""
    reference: str = getattr(row, "reference", "") or ""

    # Infer domain from reference_contexts metadata if available
    contexts = getattr(row, "reference_contexts", []) or []
    domain = "unknown"
    if contexts:
        first_ctx = contexts[0]
        if isinstance(first_ctx, str):
            # Try to extract from content prefix pattern "ww1_"
            for tag in ("ww1", "ww2", "figures", "revolutions"):
                if tag in first_ctx[:200].lower():
                    domain = tag
                    break

    entry_id = f"gd_{idx + 1:03d}"

    return {
        "id": entry_id,
        "domain": domain,
        "question_type": our_type,
        "difficulty": "medium",          # Placeholder — SME review may update
        "question": question,
        "reference_answer": reference,
        "expected_chunks": [],           # Populated manually or via retrieval run
        "checklist_flags": {
            "faith_no_hallucination": True,
            "faith_no_contradiction": True,
            "faith_temporal_accuracy": True,
            "faith_numeric_fidelity": True,
            "faith_proper_naming": True,
            "relevancy_addresses_question": True,
            "relevancy_no_tangent": True,
            "relevancy_intent_match": True,
            "completeness_all_parts": True,
            "completeness_no_omission": True,
            "precision_chunks_relevant": True,
            "recall_no_gaps": True,
        },
        "needs_human_review": False,
        "sme1_verdict": "",
        "sme2_verdict": "",
        "reviewer_notes": "",
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_generation(use_saved_kg: bool = False) -> None:
    """
    Full pipeline: load KB → build/load KG → generate testset → save outputs.
    """
    from openai import OpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator

    # ------------------------------------------------------------------ #
    # Credentials & model clients
    # ------------------------------------------------------------------ #
    credentials = load_vertex_credentials()

    # AnthropicVertex is sync — RAGAS generator is synchronous internally.
    from anthropic import AnthropicVertex

    claude_client = AnthropicVertex(
        project_id=settings.vertex_project_id,
        region=settings.vertex_region,
        credentials=credentials,
    )
    claude_llm = llm_factory(
        settings.claude_model,
        provider="anthropic",
        client=claude_client,
    )

    openai_client = OpenAI(api_key=settings.openai_api_key)
    embeddings = embedding_factory(
        settings.embedding_model,
        provider="openai",
        client=openai_client,
    )

    # ------------------------------------------------------------------ #
    # Knowledge graph
    # ------------------------------------------------------------------ #
    if use_saved_kg:
        kg = load_knowledge_graph()
    else:
        docs = load_documents()
        kg = build_knowledge_graph(docs, claude_llm, embeddings)

    # ------------------------------------------------------------------ #
    # Testset generation
    # ------------------------------------------------------------------ #
    logger.info("Generating testset (target: 100 Q&A pairs)...")

    generator = TestsetGenerator(
        llm=claude_llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
    )

    # Use default distribution — RAGAS will spread across synthesizer types.
    # For precise control over type counts, post-process by filtering.
    try:
        from ragas.testset.synthesizers import default_query_distribution
        distribution = default_query_distribution(generator.llm)
    except ImportError:
        distribution = None  # RAGAS will use its own default

    if distribution is not None:
        dataset = generator.generate(testset_size=100, query_distribution=distribution)
    else:
        dataset = generator.generate(testset_size=100)

    logger.info("Generation complete. Formatting %d samples.", len(dataset))

    # ------------------------------------------------------------------ #
    # Format entries
    # ------------------------------------------------------------------ #
    entries = [_format_entry(i, row) for i, row in enumerate(dataset.samples)]

    # ------------------------------------------------------------------ #
    # Save raw output (NEVER overwrite — audit trail)
    # ------------------------------------------------------------------ #
    if _RAW_PATH.exists():
        logger.warning(
            "raw_generated.json already exists at %s — skipping raw save "
            "(delete the file manually to regenerate).",
            _RAW_PATH,
        )
    else:
        raw_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(entries),
            "entries": entries,
        }
        _RAW_PATH.write_text(
            json.dumps(raw_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Raw generated dataset saved to %s.", _RAW_PATH)

    # ------------------------------------------------------------------ #
    # Save working copy (always overwritten)
    # ------------------------------------------------------------------ #
    golden_payload = {
        "version": "1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(entries),
        "entries": entries,
    }
    _GOLDEN_PATH.write_text(
        json.dumps(golden_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Golden dataset (%d entries) saved to %s.", len(entries), _GOLDEN_PATH
    )

    print(
        f"\nDone. {len(entries)} Q&A pairs saved.\n"
        f"  Raw (immutable): {_RAW_PATH}\n"
        f"  Working copy:    {_GOLDEN_PATH}\n"
        "\nNote: adversarial (8+) and ambiguous (5+) questions must be added manually.\n"
        "Run golden_dataset/sme_review.py next to apply AI SME review."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate golden dataset Q&A pairs using RAGAS TestsetGenerator."
    )
    parser.add_argument(
        "--use-saved-kg",
        action="store_true",
        help="Load the previously-saved knowledge graph instead of rebuilding it.",
    )
    args = parser.parse_args()

    try:
        run_generation(use_saved_kg=args.use_saved_kg)
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        sys.exit(1)
