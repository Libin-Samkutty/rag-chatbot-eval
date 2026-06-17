"""
ui/pages/04_history.py — Run history page.

Calls GET /api/history via httpx. Never imports from routers/ directly.
"""

import httpx
import pandas as pd
import streamlit as st

from config import settings

API_BASE = f"http://localhost:{settings.api_port}"

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

st.title("Run History")

limit = st.sidebar.slider("Rows to load", min_value=10, max_value=100, value=50, step=10)

try:
    resp = httpx.get(
        f"{API_BASE}/api/history",
        params={"limit": limit},
        timeout=30.0,
    )
    resp.raise_for_status()
    records = resp.json()
except httpx.HTTPStatusError as exc:
    st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    st.stop()
except httpx.RequestError as exc:
    st.error(f"Could not reach the API at {API_BASE}. Is the server running? Details: {exc}")
    st.stop()

if not records:
    st.info("No history yet. Send a question in the Chat page first.")
    st.stop()

df = pd.DataFrame(records)

# Truncate question for the summary table
TABLE_COLS = [
    "created_at",
    "question",
    "domain_tag",
    "overall_passed",
    "latency_ms",
    "faith_passed",
    "answer_relevancy_passed",
    "completeness_passed",
]

available_cols = [c for c in TABLE_COLS if c in df.columns]

display_df = df[available_cols].copy()
if "question" in display_df.columns:
    display_df["question"] = display_df["question"].str[:80] + "..."

st.subheader(f"Last {len(display_df)} runs")
st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.subheader("Detailed View")

for i, row in df.iterrows():
    q_short = str(row.get("question", ""))[:80]
    passed = row.get("overall_passed", False)
    icon = "PASS" if passed else "FAIL"
    domain = row.get("domain_tag", "general")

    with st.expander(f"[{icon}] {q_short}... ({domain})", expanded=False):
        st.markdown(f"**Question:** {row.get('question', '')}")
        st.markdown(f"**Answer:** {row.get('answer', '')}")
        st.caption(
            f"ID: {row.get('id', '')} | "
            f"Created: {row.get('created_at', '')} | "
            f"Latency: {row.get('latency_ms', 0):.0f} ms"
        )

        checklist = row.get("checklist_json")
        if checklist and isinstance(checklist, dict):
            st.markdown("**Full Eval Detail:**")
            DIMS = [
                "faithfulness", "answer_relevancy", "completeness",
                "context_precision", "context_recall",
                "coherence", "historical_balance", "toxicity",
            ]
            for dim_key in DIMS:
                dim = checklist.get(dim_key, {})
                if not dim:
                    continue
                dim_passed = dim.get("passed", False)
                dim_icon = "PASS" if dim_passed else "FAIL"
                label = dim_key.replace("_", " ").title()
                st.markdown(
                    f"- **{label}** [{dim_icon}] — {dim.get('reason', '')}"
                )
        elif checklist:
            st.json(checklist)
