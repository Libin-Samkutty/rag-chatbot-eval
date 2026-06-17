"""
rag/indexer.py — Indexes the knowledge base .txt files into ChromaDB.

Runs once on startup (via main.py lifespan). Skips indexing if the collection
already has documents. Delete ./chroma_db/ to force a full re-index.

ChromaDB stores per chunk:
  - The chunk text (document)
  - An embedding from text-embedding-3-small
  - Metadata: source, chunk_index, domain_tag
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chromadb
from openai import AsyncOpenAI

from config import settings
from rag.chunker import chunk_file

logger = logging.getLogger(__name__)

_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


async def _embed_batch(texts: list[str], client: AsyncOpenAI) -> list[list[float]]:
    response = await client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


async def index_knowledge_base() -> None:
    """
    Index all .txt files from the knowledge directory into ChromaDB.

    Skips if the collection is already populated. Delete ./chroma_db/ to
    force a full re-index (required after changing chunk size or KB files).
    """
    collection = get_collection()

    existing_count = collection.count()
    if existing_count > 0:
        logger.info(
            "ChromaDB collection '%s' already has %d documents. Skipping indexing.",
            settings.collection_name,
            existing_count,
        )
        return

    knowledge_dir = Path(settings.knowledge_path)
    txt_files = list(knowledge_dir.glob("*.txt"))

    if not txt_files:
        logger.warning(
            "No .txt files found in '%s'. The chatbot will have no knowledge base.",
            settings.knowledge_path,
        )
        return

    logger.info("Indexing %d files from '%s'...", len(txt_files), settings.knowledge_path)

    all_chunks: list[dict] = []
    for path in sorted(txt_files):
        chunks = chunk_file(path)
        all_chunks.extend(chunks)
        logger.info("  %s → %d chunks (domain: %s)", path.name, len(chunks), chunks[0]["domain_tag"])

    logger.info("Total chunks to index: %d", len(all_chunks))

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    batch_size = 100

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = await _embed_batch(texts, openai_client)

        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            global_index = i + j
            all_ids.append(f"chunk_{global_index}")
            all_embeddings.append(embedding)
            all_documents.append(chunk["text"])
            all_metadatas.append({
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "domain_tag": chunk["domain_tag"],
            })

        logger.info(
            "  Embedded batch %d/%d",
            i // batch_size + 1,
            (len(all_chunks) + batch_size - 1) // batch_size,
        )

    collection.upsert(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    logger.info("Indexed %d chunks into ChromaDB.", len(all_chunks))
