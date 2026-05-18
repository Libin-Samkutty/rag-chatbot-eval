"""
rag/retriever.py — Retrieves the most relevant chunks for a user query.

Steps:
  1. Embed the user's query using text-embedding-3-small
  2. Query ChromaDB for the top-k most similar chunks (by cosine distance)
  3. Return the chunk texts and their source filenames

The retriever is the component most worth experimenting with:
  - Increase TOP_K_CHUNKS to give the model more context (but higher cost)
  - Decrease it to force the model to be more selective (but risk missing info)
  - Try different embedding models to see how retrieval quality changes
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
) -> tuple[list[str], list[str]]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:   The user's question.
        client:  An async OpenAI client instance.
        top_k:   Number of chunks to retrieve. Defaults to settings.top_k_chunks.

    Returns:
        A tuple of (chunk_texts, chunk_sources):
          - chunk_texts:   List of raw text strings, ordered by relevance.
          - chunk_sources: Corresponding source filenames (e.g. 'transformer.txt').
    """
    if top_k is None:
        top_k = settings.top_k_chunks

    # Step 1: Embed the query
    embedding_response = await client.embeddings.create(
        input=query,
        model=settings.embedding_model,
    )
    query_embedding = embedding_response.data[0].embedding

    # Step 2: Query ChromaDB
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # results["documents"] is a list of lists (one per query — we sent one)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    chunk_texts = documents
    chunk_sources = [meta.get("source", "unknown") for meta in metadatas]

    logger.debug(
        "Retrieved %d chunks for query: '%s...'",
        len(chunk_texts),
        query[:60],
    )

    return chunk_texts, chunk_sources
