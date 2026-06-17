"""
ui/app.py — Streamlit multi-page app entry point.

Run with:
    streamlit run ui/app.py --server.port 8501

The four pages are in ui/pages/ and are auto-discovered by Streamlit.
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Eval Harness",
    page_icon="magnifying glass",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS matching IBM Plex font, #0f1117 background
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap');

:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e2e4ef;
    --accent: #4f8ef7;
    --pass: #22c55e;
    --fail: #ef4444;
    --warn: #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

code, pre, [data-testid="stCode"] {
    font-family: 'IBM Plex Mono', monospace;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text);
}

.stButton > button {
    background-color: var(--accent);
    color: #fff;
    border: none;
    border-radius: 4px;
}

.stButton > button:hover {
    background-color: #3a7bd5;
}

[data-testid="stExpander"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("RAG Eval Harness")
st.sidebar.markdown("World History knowledge base with 8-dimension eval.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Pages**
- Chat
- Eval Dashboard
- Golden Review
- History
"""
)

st.write("## Welcome to the RAG Eval Harness")
st.write(
    "This tool demonstrates an educational RAG evaluation framework. "
    "Every chat turn runs **eight evaluation dimensions** and surfaces the scores "
    "so you can inspect exactly why each turn passed or failed."
)

st.info(
    "Use the sidebar to navigate to **Chat**, **Eval Dashboard**, "
    "**Golden Review**, or **History**."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Eval Dimensions", "8")
with col2:
    st.metric("Knowledge Domains", "4")
with col3:
    st.metric("Eval Model", "gpt-4o")

st.markdown("---")
st.markdown(
    """
**Architecture notes**
- Evals: 4 RAGAS-inspired metrics via direct GPT-4o async judge calls; 4 holistic dimensions via DeepEval G-Eval.
- Generation: Gemini 2.5 Flash via Vertex AI.
- Judge: GPT-4o via OpenAI.
- Embeddings: text-embedding-3-small via OpenAI.
- Knowledge base: World History (WW1, WW2, Historical Figures, Revolutions).
"""
)
