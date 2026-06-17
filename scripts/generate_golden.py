"""
scripts/generate_golden.py — CLI entry point for golden dataset generation.

Calls golden_dataset/generator.py to produce the golden dataset using
RAGAS knowledge graph synthesis + Claude (via Vertex AI) as the gold
answer generator.

Usage:
    python scripts/generate_golden.py
    python scripts/generate_golden.py --use-saved-kg
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from golden_dataset.generator import generate_golden_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate golden dataset using RAGAS + Claude"
    )
    parser.add_argument(
        "--use-saved-kg",
        action="store_true",
        help="Reload a previously saved knowledge graph instead of rebuilding it",
    )
    args = parser.parse_args()

    generate_golden_dataset(use_saved_kg=args.use_saved_kg)


if __name__ == "__main__":
    main()
