"""
ui/pages/01_chat.py — Chat interface page.

Calls POST /api/chat via httpx. Never imports from routers/ directly.
"""

import httpx
import streamlit as st

from config import settings  # for API base URL only — no internal imports

API_BASE = f"http://localhost:{settings.api_port}"

# Dark theme CSS (same variables as app.py)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap');
:root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e2e4ef; --accent: #4f8ef7;
    --pass: #22c55e; --fail: #ef4444;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg); color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] { background-color: var(--surface); }
code, pre { font-family: 'IBM Plex Mono', monospace; }
.pass-badge {
    background: #166534; color: #bbf7d0; padding: 2px 10px;
    border-radius: 12px; font-weight: 600; font-size: 0.85rem;
}
.fail-badge {
    background: #7f1d1d; color: #fecaca; padding: 2px 10px;
    border-radius: 12px; font-weight: 600; font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Chat")

DOMAIN_OPTIONS = {
    "All domains": None,
    "WW1": "ww1",
    "WW2": "ww2",
    "Historical Figures": "historical_figures",
    "Revolutions": "revolutions",
}

with st.form("chat_form"):
    question = st.text_area("Your question", height=80, placeholder="Ask about world history...")
    domain_label = st.selectbox("Domain filter", list(DOMAIN_OPTIONS.keys()))
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    domain_filter = DOMAIN_OPTIONS[domain_label]

    with st.spinner("Generating answer and running evals..."):
        try:
            response = httpx.post(
                f"{API_BASE}/api/chat",
                json={"question": question, "domain_filter": domain_filter},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except httpx.RequestError as exc:
            st.error(
                f"Could not reach the API at {API_BASE}. "
                f"Is `uvicorn main:app --reload` running? Details: {exc}"
            )
            st.stop()

    # --- Answer ---
    st.markdown("### Answer")
    st.markdown(
        f"<div style='background:#1a1d27;padding:16px;border-radius:8px;"
        f"border:1px solid #2a2d3a;line-height:1.7'>{data['answer']}</div>",
        unsafe_allow_html=True,
    )

    latency = data.get("latency_ms", 0)
    st.caption(f"Latency: {latency:.0f} ms | Message ID: {data.get('message_id', '')}")

    # --- Retrieved Chunks ---
    with st.expander("Retrieved Chunks", expanded=False):
        for i, chunk in enumerate(data.get("chunks", []), start=1):
            st.markdown(
                f"**Chunk {i}** — source: `{chunk.get('source', 'unknown')}` "
                f"| domain: `{chunk.get('domain_tag', 'general')}`"
            )
            st.text(chunk.get("text", ""))
            st.markdown("---")

    # --- Eval Results ---
    st.markdown("### Eval Results")

    eval_result = data.get("eval_result", {})
    overall = eval_result.get("overall_passed", False)

    badge_html = (
        '<span class="pass-badge">OVERALL PASS</span>'
        if overall
        else '<span class="fail-badge">OVERALL FAIL</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("")

    DIMENSION_KEYS = [
        "faithfulness",
        "answer_relevancy",
        "completeness",
        "context_precision",
        "context_recall",
        "coherence",
        "historical_balance",
        "toxicity",
    ]

    for key in DIMENSION_KEYS:
        dim = eval_result.get(key, {})
        if not dim:
            continue

        passed = dim.get("passed", False)
        icon = "✓" if passed else "✗"
        color = "#22c55e" if passed else "#ef4444"
        label = key.replace("_", " ").title()

        with st.expander(f"{icon} {label}", expanded=not passed):
            st.markdown(f"**Reason:** {dim.get('reason', '')}")

            # G-Eval holistic dimensions — show score
            score = dim.get("score")
            if score is not None:
                st.markdown(f"**Score:** {score:.2f}")

            # Checklist dimensions — show each item
            items = dim.get("items", [])
            if items:
                st.markdown("**Checklist items:**")
                for item in items:
                    item_icon = "✓" if item.get("result") else "✗"
                    tier_label = f"T{item.get('tier', '?')}"
                    st.markdown(
                        f"&nbsp;&nbsp;{item_icon} `{item.get('key', '')}` "
                        f"({tier_label}) — {item.get('question', '')}"
                    )

            tier1_failed = dim.get("tier1_failed", [])
            if tier1_failed:
                st.error(f"Tier 1 failures: {', '.join(tier1_failed)}")

            rate = dim.get("tier2_pass_rate")
            if rate is not None:
                st.caption(f"Tier 2 pass rate: {rate:.0%}")

elif submitted and not question.strip():
    st.warning("Please enter a question before submitting.")
