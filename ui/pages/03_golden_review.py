"""
ui/pages/03_golden_review.py — HITL golden dataset review page.

Loads tests/evals/golden_dataset.json directly from disk. Writes back
updates when the reviewer clicks Save. No authentication — single reviewer,
local use only.
"""

import csv
import io
import json
from pathlib import Path

import streamlit as st

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap');
:root { --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a; --text: #e2e4ef; --accent: #4f8ef7; }
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stSidebar"] { background-color: var(--surface); }
code, pre { font-family: 'IBM Plex Mono', monospace; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Golden Dataset Review")

# Locate the golden dataset relative to this file (works regardless of cwd)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLDEN_PATH = _PROJECT_ROOT / "tests" / "evals" / "golden_dataset.json"


def _load_dataset() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_dataset(entries: list[dict]) -> None:
    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


entries = _load_dataset()

if not entries:
    st.warning(f"Golden dataset not found at: {GOLDEN_PATH}")
    st.stop()

total = len(entries)
flagged = sum(1 for e in entries if e.get("needs_human_review", False))

col_a, col_b = st.columns(2)
col_a.metric("Total Entries", total)
col_b.metric("Needs Human Review", flagged)

st.markdown("---")

# --- Navigation ---
if "golden_index" not in st.session_state:
    st.session_state.golden_index = 0

idx = st.session_state.golden_index
idx = max(0, min(idx, total - 1))

nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
with nav_col1:
    if st.button("Prev") and idx > 0:
        st.session_state.golden_index = idx - 1
        st.rerun()
with nav_col3:
    if st.button("Next") and idx < total - 1:
        st.session_state.golden_index = idx + 1
        st.rerun()
with nav_col2:
    st.caption(f"Entry {idx + 1} of {total}")

entry = entries[idx]

# --- Metadata chips ---
meta_parts = []
if entry.get("id"):
    meta_parts.append(f"`{entry['id']}`")
if entry.get("domain"):
    meta_parts.append(f"domain: `{entry['domain']}`")
if entry.get("question_type"):
    meta_parts.append(f"type: `{entry['question_type']}`")
if entry.get("difficulty"):
    meta_parts.append(f"difficulty: `{entry['difficulty']}`")
if meta_parts:
    st.markdown(" &nbsp;|&nbsp; ".join(meta_parts), unsafe_allow_html=True)

# --- Human review warning ---
if entry.get("needs_human_review", False):
    st.warning("This entry has been flagged for human review.")

# --- Core content ---
st.markdown(f"**Question:** {entry.get('question', '')}")

if entry.get("reference_answer"):
    with st.expander("Reference Answer", expanded=True):
        st.write(entry["reference_answer"])

if entry.get("expected_chunks"):
    with st.expander("Expected Chunks"):
        for i, chunk in enumerate(entry["expected_chunks"], start=1):
            st.markdown(f"{i}. {chunk}")

# SME verdicts
sme1 = entry.get("sme1_verdict")
sme2 = entry.get("sme2_verdict")
if sme1 or sme2:
    v_col1, v_col2 = st.columns(2)
    if sme1:
        color = "#22c55e" if sme1 == "pass" else "#ef4444"
        v_col1.markdown(
            f"<span style='color:{color};font-weight:600'>SME1: {sme1.upper()}</span>",
            unsafe_allow_html=True,
        )
    if sme2:
        color = "#22c55e" if sme2 == "pass" else "#ef4444"
        v_col2.markdown(
            f"<span style='color:{color};font-weight:600'>SME2: {sme2.upper()}</span>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# --- Per-dimension checklist flags ---
checklist_flags: dict = entry.get("checklist_flags", {})
updated_flags: dict = {}

if checklist_flags:
    st.markdown("**Checklist Flags**")
    for key, current_val in checklist_flags.items():
        updated_flags[key] = st.checkbox(
            key.replace("_", " ").title(),
            value=bool(current_val),
            key=f"flag_{idx}_{key}",
        )

# --- Reviewer notes ---
reviewer_notes = st.text_area(
    "Reviewer Notes",
    value=entry.get("reviewer_notes", ""),
    height=100,
    key=f"notes_{idx}",
)

needs_review = st.checkbox(
    "Flag for human review",
    value=bool(entry.get("needs_human_review", False)),
    key=f"review_{idx}",
)

# --- Save ---
if st.button("Save Changes"):
    entries[idx] = {
        **entry,
        "checklist_flags": updated_flags if checklist_flags else entry.get("checklist_flags", {}),
        "reviewer_notes": reviewer_notes,
        "needs_human_review": needs_review,
    }
    try:
        _save_dataset(entries)
        st.success("Saved.")
    except OSError as exc:
        st.error(f"Could not write to {GOLDEN_PATH}: {exc}")

st.markdown("---")

# --- Export CSV ---
st.markdown("**Export**")
if st.button("Export all entries to CSV"):
    output = io.StringIO()
    if entries:
        flat_rows = []
        for e in entries:
            row = {k: v for k, v in e.items() if not isinstance(v, (dict, list))}
            # Flatten expected dict
            expected = e.get("expected", {})
            for ek, ev in expected.items():
                row[f"expected_{ek}"] = ev
            flat_rows.append(row)

        fieldnames = list({k for r in flat_rows for k in r.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    csv_bytes = output.getvalue().encode("utf-8")
    st.download_button(
        label="Download golden_dataset_export.csv",
        data=csv_bytes,
        file_name="golden_dataset_export.csv",
        mime="text/csv",
    )
