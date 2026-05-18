"""
rag/chunker.py — Splits plain text into overlapping, token-bounded chunks.

Why chunk?
  Language models have a context window limit, and embedding models work best
  on short, focused passages. We split each .txt article into chunks of ~400
  tokens with a 50-token overlap so that information near chunk boundaries
  isn't lost.

Why tiktoken?
  tiktoken is OpenAI's tokenizer — it counts tokens the same way the embedding
  model does, so we get accurate chunk sizes rather than rough word estimates.

Chunk size choices:
  - 400 tokens ≈ 300 words ≈ 2-3 paragraphs — enough context for an answer
  - 50-token overlap — prevents cutting sentences/ideas at boundaries
"""

from pathlib import Path

import tiktoken


# The cl100k_base encoding is used by text-embedding-3-small and gpt-4o-mini
ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 400    # tokens
CHUNK_OVERLAP = 50  # tokens


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping token-bounded chunks.

    Args:
        text:   The full text of one article.
        source: The filename (used as metadata in ChromaDB).

    Returns:
        List of dicts with keys: 'text', 'source', 'chunk_index'
    """
    # Tokenise the full text
    tokens = ENCODING.encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        # Decode back to a string for storage
        chunk_text_str = ENCODING.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        # Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) to create the overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_file(path: Path) -> list[dict]:
    """Read a .txt file and return its chunks."""
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, source=path.name)
