"""
rag/retriever.py — Retrieves the most relevant chunks for a user query.

Steps:
  1. Embed the query with text-embedding-3-small
  2. Query ChromaDB (cosine similarity), optionally filtered by domain_tag
  3. Return chunk dicts with text, source, domain_tag
"""

import logging

from openai import AsyncOpenAI

from config import settings
from rag.indexer import get_collection

logger = logging.getLogger(__name__)


async def retrieve_chunks(
    query: str,
    client: AsyncOpenAI,
    top_k: int | None = None,
    domain_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:         The user's question.
        client:        An async OpenAI client instance.
        top_k:         Number of chunks to retrieve (defaults to settings.top_k_chunks).
        domain_filter: If set, restricts results to chunks with this domain_tag
                       (e.g. "ww1", "ww2", "historical_figures", "revolutions").

    Returns:
        List of dicts with keys: text, source, domain_tag — ordered by relevance.
    """
    if top_k is None:
        top_k = settings.top_k_chunks

    embedding_response = await client.embeddings.create(
        input=query,
        model=settings.embedding_model,
    )
    query_embedding = embedding_response.data[0].embedding

    collection = get_collection()

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if domain_filter is not None:
        query_kwargs["where"] = {"domain_tag": domain_filter}

    results = collection.query(**query_kwargs)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    chunks = [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "domain_tag": meta.get("domain_tag", "general"),
        }
        for doc, meta in zip(documents, metadatas)
    ]

    logger.debug("Retrieved %d chunks for query: '%s...'", len(chunks), query[:60])
    return chunks
