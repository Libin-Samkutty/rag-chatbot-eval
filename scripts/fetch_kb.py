"""
scripts/fetch_kb.py — Fetches Wikipedia articles for the World History knowledge base.

Files are saved with a domain-tag prefix so the RAG chunker can derive domain_tag
from the filename:  ww1_*, ww2_*, figures_*, revolutions_*

Usage:
    pip install wikipedia-api  # extra dependency, not in requirements.txt
    python scripts/fetch_kb.py
    python scripts/fetch_kb.py --refresh  # overwrites existing files
"""

import argparse
import sys
import time
from pathlib import Path

# Each entry: (wikipedia_title, output_filename)
# Filename convention: {domain_tag}_{slug}.txt
ARTICLES: list[tuple[str, str]] = [
    # WW1
    ("World War I",                                  "ww1_world_war_i.txt"),
    ("Assassination of Archduke Franz Ferdinand",    "ww1_assassination_franz_ferdinand.txt"),
    ("Trench warfare",                               "ww1_trench_warfare.txt"),
    ("Treaty of Versailles",                         "ww1_treaty_of_versailles.txt"),
    ("Western Front (World War I)",                  "ww1_western_front.txt"),
    ("Eastern Front (World War I)",                  "ww1_eastern_front.txt"),
    ("Ottoman Empire",                               "ww1_ottoman_empire_wwi.txt"),

    # WW2
    ("World War II",                                 "ww2_world_war_ii.txt"),
    ("The Holocaust",                                "ww2_holocaust.txt"),
    ("Normandy landings",                            "ww2_d_day.txt"),
    ("Battle of Stalingrad",                         "ww2_battle_of_stalingrad.txt"),
    ("Atomic bombings of Hiroshima and Nagasaki",    "ww2_atomic_bombings_hiroshima_nagasaki.txt"),
    ("Nazi Germany",                                 "ww2_nazi_germany.txt"),
    ("Pacific War",                                  "ww2_pacific_war.txt"),

    # Historical Figures
    ("Napoleon",                                     "figures_napoleon_bonaparte.txt"),
    ("Winston Churchill",                            "figures_winston_churchill.txt"),
    ("Adolf Hitler",                                 "figures_adolf_hitler.txt"),
    ("Joseph Stalin",                                "figures_joseph_stalin.txt"),
    ("Mahatma Gandhi",                               "figures_mahatma_gandhi.txt"),
    ("Nelson Mandela",                               "figures_nelson_mandela.txt"),
    ("Vladimir Lenin",                               "figures_vladimir_lenin.txt"),

    # Revolutions and Independence Movements
    ("French Revolution",                            "revolutions_french_revolution.txt"),
    ("American Revolution",                          "revolutions_american_revolution.txt"),
    ("Russian Revolution",                           "revolutions_russian_revolution.txt"),
    ("Indian independence movement",                 "revolutions_indian_independence.txt"),
    ("Chinese Communist Revolution",                 "revolutions_chinese_revolution.txt"),
    ("Haitian Revolution",                           "revolutions_haitian_revolution.txt"),
    ("Cuban Revolution",                             "revolutions_cuban_revolution.txt"),
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

    for title, filename in ARTICLES:
        output_path = OUTPUT_DIR / filename

        if output_path.exists() and not refresh:
            print(f"  SKIP  {filename} (already exists)")
            skip_count += 1
            continue

        page = wiki.page(title)

        if not page.exists():
            print(f"  FAIL  '{title}' — page not found on Wikipedia")
            fail_count += 1
            continue

        content = f"# {page.title}\n\n{page.text}"
        output_path.write_text(content, encoding="utf-8")
        print(f"  OK    {filename} ({len(content):,} chars)")
        success_count += 1

        time.sleep(0.5)

    print(f"\nDone: {success_count} fetched, {skip_count} skipped, {fail_count} failed")
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
