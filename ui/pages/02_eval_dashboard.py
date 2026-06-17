"""
ui/pages/02_eval_dashboard.py — Eval metrics dashboard.

Calls GET /api/eval-runs via httpx. Never imports from routers/ directly.
"""

from datetime import date, timedelta

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

st.title("Eval Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")
DOMAIN_OPTIONS = ["All", "ww1", "ww2", "historical_figures", "revolutions"]
selected_domain = st.sidebar.selectbox("Domain", DOMAIN_OPTIONS)

today = date.today()
date_from = st.sidebar.date_input("From", value=today - timedelta(days=30))
date_to = st.sidebar.date_input("To", value=today)
limit = st.sidebar.slider("Max runs to fetch", min_value=10, max_value=200, value=100, step=10)

# --- Fetch data ---
params: dict = {"limit": limit, "offset": 0}
if selected_domain != "All":
    params["domain"] = selected_domain

try:
    resp = httpx.get(f"{API_BASE}/api/eval-runs", params=params, timeout=30.0)
    resp.raise_for_status()
    payload = resp.json()
    runs = payload.get("runs", [])
except httpx.HTTPStatusError as exc:
    st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    st.stop()
except httpx.RequestError as exc:
    st.error(f"Could not reach the API at {API_BASE}. Is the server running? Details: {exc}")
    st.stop()

if not runs:
    st.info("No eval runs found. Send a few questions in the Chat page first.")
    st.stop()

df = pd.DataFrame(runs)

# Parse and filter by date range
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    mask = (df["created_at"].dt.date >= date_from) & (df["created_at"].dt.date <= date_to)
    df = df[mask]

if df.empty:
    st.info("No runs in the selected date range.")
    st.stop()

# --- Summary stats ---
DIMENSION_PASS_COLS = [
    "faith_passed",
    "answer_relevancy_passed",
    "completeness_passed",
    "context_recall_passed",
    "coherence_passed",
    "historical_balance_passed",
    "toxicity_passed",
]

total_runs = len(df)
overall_pass_rate = df["overall_passed"].mean() * 100 if "overall_passed" in df.columns else 0.0
avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0

# Average faithfulness pass rate
faith_col = "faith_passed"
avg_faith = df[faith_col].mean() * 100 if faith_col in df.columns else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Runs", total_runs)
col2.metric("Overall Pass Rate", f"{overall_pass_rate:.1f}%")
col3.metric("Avg Faithfulness Pass", f"{avg_faith:.1f}%")
col4.metric("Avg Latency", f"{avg_latency:.0f} ms")

st.markdown("---")

# --- Overall pass rate over last N runs ---
st.subheader("Overall Pass Rate Over Time")
if "overall_passed" in df.columns and "created_at" in df.columns:
    time_df = (
        df.sort_values("created_at")
        .reset_index(drop=True)
        .assign(run_number=lambda x: range(1, len(x) + 1))
    )
    # Rolling 10-run average
    time_df["rolling_pass_rate"] = (
        time_df["overall_passed"].astype(float).rolling(window=10, min_periods=1).mean()
    )
    st.line_chart(time_df.set_index("run_number")[["rolling_pass_rate"]])
else:
    st.info("Not enough data for a time series chart.")

st.markdown("---")

# --- Pass rate per dimension ---
st.subheader("Pass Rate per Dimension")
available_cols = [c for c in DIMENSION_PASS_COLS if c in df.columns]
if available_cols:
    pass_rates = df[available_cols].mean() * 100
    pass_rates.index = [c.replace("_passed", "").replace("_", " ").title() for c in pass_rates.index]
    st.bar_chart(pass_rates)
else:
    st.info("Dimension pass columns not available in the data.")

st.markdown("---")

# --- Raw data table ---
with st.expander("Raw Runs Table", expanded=False):
    display_cols = [
        c for c in ["created_at", "question", "domain_tag", "overall_passed", "latency_ms"]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True)
