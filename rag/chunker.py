"""
rag/chunker.py — Splits plain text into overlapping, token-bounded chunks.

Chunk size: 512 tokens with 50-token overlap (historical prose benefits from
larger windows compared to shorter technical text).

domain_tag is derived from the filename prefix:
  ww1_*         → "ww1"
  ww2_*         → "ww2"
  figures_*     → "historical_figures"
  revolutions_* → "revolutions"
  (anything else → "general")
"""

from pathlib import Path

import tiktoken

# cl100k_base is used by text-embedding-3-small and gpt-4o
ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 512    # tokens
CHUNK_OVERLAP = 50  # tokens

_DOMAIN_PREFIXES: dict[str, str] = {
    "ww1_": "ww1",
    "ww2_": "ww2",
    "figures_": "historical_figures",
    "revolutions_": "revolutions",
}


def _derive_domain_tag(filename: str) -> str:
    for prefix, tag in _DOMAIN_PREFIXES.items():
        if filename.startswith(prefix):
            return tag
    return "general"


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping token-bounded chunks.

    Args:
        text:   Full article text.
        source: Filename used as metadata in ChromaDB.

    Returns:
        List of dicts with keys: text, source, chunk_index, domain_tag
    """
    domain_tag = _derive_domain_tag(source)
    tokens = ENCODING.encode(text)

    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_str = ENCODING.decode(chunk_tokens)

        chunks.append({
            "text": chunk_str,
            "source": source,
            "chunk_index": chunk_index,
            "domain_tag": domain_tag,
        })

        chunk_index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_file(path: Path) -> list[dict]:
    """Read a .txt file and return its chunks."""
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, source=path.name)
