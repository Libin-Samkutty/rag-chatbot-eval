"""
scripts/fetch_kb.py — Fetches Wikipedia articles for the knowledge base.

This script is optional — the repo ships with pre-fetched .txt files in
knowledge/. Run this only if you want to refresh the knowledge base with
updated Wikipedia content.

Usage:
    pip install wikipedia-api  # extra dependency, not in requirements.txt
    python scripts/fetch_kb.py
    python scripts/fetch_kb.py --refresh  # overwrites existing files
"""

import argparse
import sys
import time
from pathlib import Path

# ML/AI articles to fetch — add or remove as needed
ARTICLE_TITLES = [
    "Transformer (deep learning architecture)",
    "Attention (machine learning)",
    "Retrieval-augmented generation",
    "Large language model",
    "Gradient descent",
    "Backpropagation",
    "Reinforcement learning from human feedback",
    "Word embedding",
    "Fine-tuning (deep learning)",
    "Generative pre-trained transformer",
    "BERT (language model)",
    "Recurrent neural network",
    "Long short-term memory",
    "Convolutional neural network",
    "Overfitting",
    "Regularization (mathematics)",
    "Batch normalization",
    "Dropout (neural networks)",
    "Tokenization (data security)",
    "Prompt engineering",
    "Few-shot learning (natural language processing)",
    "Zero-shot learning",
    "In-context learning (natural language processing)",
    "Chain-of-thought prompting",
    "Hallucination (artificial intelligence)",
    "Vector database",
    "Semantic search",
    "Named-entity recognition",
    "Sentiment analysis",
    "Question answering (computer science)",
    "Cosine similarity",
    "K-nearest neighbors algorithm",
    "Principal component analysis",
    "Variational autoencoder",
    "Generative adversarial network",
    "Diffusion model",
    "Neural network (machine learning)",
    "Perceptron",
    "Sigmoid function",
    "Softmax function",
    "Cross-entropy",
    "Loss function",
    "Stochastic gradient descent",
    "Adam (optimization algorithm)",
    "Learning rate",
    "Hyperparameter (machine learning)",
    "Bias–variance tradeoff",
    "Confusion matrix",
    "Precision and recall",
    "F-score",
    "Benchmark (computing)",
    "Transfer learning",
    "Self-supervised learning",
    "Contrastive learning",
    "Mixture of experts",
    "Sparse attention",
    "Positional encoding",
    "Layer normalization",
    "Encoder–decoder architecture",
    "Beam search",
    "Temperature (statistics)",
]


OUTPUT_DIR = Path(__file__).parent.parent / "knowledge"


def fetch_articles(refresh: bool = False) -> None:
    """Download Wikipedia articles and save them as .txt files."""
    try:
        import wikipediaapi
    except ImportError:
        print("Error: wikipedia-api is not installed.")
        print("Run: pip install wikipedia-api")
        sys.exit(1)

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="eval-chatbot-kb-builder/1.0 (educational project)",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for title in ARTICLE_TITLES:
        # Create a safe filename from the article title
        safe_name = title.lower().replace(" ", "_").replace("/", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        output_path = OUTPUT_DIR / f"{safe_name}.txt"

        if output_path.exists() and not refresh:
            print(f"  SKIP  {output_path.name} (already exists)")
            skip_count += 1
            continue

        page = wiki.page(title)

        if not page.exists():
            print(f"  FAIL  '{title}' — page not found on Wikipedia")
            fail_count += 1
            continue

        # Write the full article text (Wikipedia API strips most markup)
        content = f"# {page.title}\n\n{page.text}"
        output_path.write_text(content, encoding="utf-8")
        print(f"  OK    {output_path.name} ({len(content):,} chars)")
        success_count += 1

        # Be polite — don't hammer the Wikipedia API
        time.sleep(0.5)

    print(f"\n✅ Done: {success_count} fetched, {skip_count} skipped, {fail_count} failed")
    print(f"   Output: {OUTPUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. Delete ./chroma_db/ to force re-indexing")
    print("  2. Restart the server: uvicorn main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles for the knowledge base")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing .txt files",
    )
    args = parser.parse_args()
    fetch_articles(refresh=args.refresh)
