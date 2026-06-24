# RAG Eval Chatbot

A localhost chatbot that makes AI evaluations visible. Every response shows
**what was retrieved**, **what the model answered**, and **how the answer
scored** across eight evaluation dimensions — with a pass/fail checklist
surfaced in the UI for each one.

Built for developers and AI enthusiasts who want to understand why evals
matter, without the abstraction of a heavyweight eval framework.

---

## What It Demonstrates

| Concept | How it's shown |
|---|---|
| Evals are just functions | Every metric is a plain async Python function you can read in `eval/` |
| Retrieval quality affects generation | The "Retrieved Chunks" panel shows exactly what context the model used |
| Hallucination detection | Faithfulness checklist fails when the model goes beyond the retrieved context |
| LLM-as-judge pattern | Each eval dimension uses GPT-4o to judge the chatbot's (Gemini) output |
| Two-tier pass/fail | Tier 1 items are hard gates; Tier 2 items use a pass-rate threshold |
| Golden dataset testing | `tests/evals/` shows how to run batch evals like a CI pipeline |
| Lexical regression | BLEU, ROUGE-L, BERTScore run on the golden dataset with zero API calls |
| Adversarial red-teaming | `test_adversarial.py` tests jailbreaks, hallucination bait, prompt injection |
| CI/CD pipeline | GitHub Actions: Tier 1 (unit + lexical, no secrets) + Tier 2 (full eval suite) |
| Experiment tracking | MLflow logs 14 metrics per eval run when `MLFLOW_TRACKING_URI` is set |

---

## Requirements

- Python 3.11+
- OpenAI API key (`gpt-4o` judge + `text-embedding-3-small`)
- Google Cloud project with Vertex AI enabled (Gemini 2.5 Flash for chat)
- Service account JSON at `./credentials/client_secrets.json`

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd rag-chatbot-eval

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Set OPENAI_API_KEY and VERTEX_PROJECT_ID in .env

# 5. Fetch and index the knowledge base (one-time)
python scripts/fetch_kb.py
```

---

## Running

```bash
# Terminal 1 — FastAPI backend
uvicorn main:app --reload

# Terminal 2 — Streamlit frontend
streamlit run ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

On first startup, the backend indexes 28 World History Wikipedia articles into
ChromaDB (~1,000 chunks). This takes about 30 seconds and only happens once.

---

## Running the Eval Test Suite

```bash
# Unit tests only (no API calls)
pytest tests/ -v

# Full golden dataset suite (makes real OpenAI calls, ~$0.05)
pytest tests/evals/ --run-evals -v

# Single domain only
pytest tests/evals/ --run-evals --domain ww1 -v
```

---

## Project Structure

```
rag-chatbot-eval/
├── main.py                  # FastAPI app, startup indexing
├── config.py                # Environment settings + Vertex AI credentials
├── database.py              # SQLite helpers
├── routers/                 # chat, history, eval_runs API routes
├── rag/                     # Chunking (512 tokens), indexing, retrieval
├── eval/
│   ├── models.py            # ChecklistItem, DimensionResult, EvalResult, LexicalResult
│   ├── runner.py            # asyncio.gather over 8 dimensions + MLflow logging
│   ├── ragas_eval.py        # 4 retrieval-linked dimensions (direct GPT-4o)
│   ├── deepeval_eval.py     # 4 holistic dimensions (DeepEval G-Eval)
│   ├── lexical_eval.py      # BLEU, ROUGE-L, BERTScore (golden dataset only, no API)
│   └── checklists/          # Per-dimension pass/fail logic
├── golden_dataset/          # RAGAS TestsetGenerator + SME review pipeline
├── knowledge/               # 28 World History Wikipedia articles (.txt)
├── scripts/                 # fetch_kb.py, generate_golden.py
├── ui/                      # Streamlit multi-page app
├── .github/workflows/       # ci.yml (Tier 1, no API) + eval-suite.yml (Tier 2, nightly)
└── tests/                   # Unit tests + golden dataset eval suite + adversarial suite
```

---

## The Eight Eval Dimensions

| Dimension | Framework | What it checks |
|---|---|---|
| Faithfulness | Direct GPT-4o | Answer claims are grounded in retrieved context |
| Answer Relevancy | Direct GPT-4o | Answer addresses the question intent |
| Completeness | DeepEval G-Eval | Answer covers all parts of the question |
| Context Precision | Direct GPT-4o | Retrieved chunks were actually useful |
| Context Recall | Direct GPT-4o | Retrieved context covers the ground-truth claims |
| Coherence | DeepEval G-Eval | Answer is logically structured and readable |
| Historical Balance | DeepEval G-Eval | Contested topics are presented without bias |
| Toxicity | DeepEval G-Eval | Answer contains no harmful content (hard gate) |

Each dimension produces a checklist of Tier 1 (hard gate) and Tier 2
(threshold) items. A dimension passes only if all Tier 1 items pass and
enough Tier 2 items meet their threshold.

---

## Design Note: Why Not RAGAS?

The original implementation used the RAGAS library for the five retrieval-linked
dimensions. We replaced it with direct `AsyncOpenAI` judge calls for two reasons:

1. **Broken dependency**: RAGAS internally imports
   `langchain_community.chat_models.vertexai` during initialisation for every
   metric — even when using the OpenAI provider. That module was removed from
   `langchain_community` in versions ≥ 0.3, causing a `ModuleNotFoundError` at
   runtime with no clean workaround short of pinning a stale LangChain version.

2. **Transparency**: RAGAS wraps LLM judge calls in a LangChain pipeline that
   obscures what prompt is actually being sent. Our direct GPT-4o prompts are
   in plain sight in `eval/ragas_eval.py` — consistent with the project's goal
   of making every eval step readable and inspectable.

The checklist system, pass/fail thresholds, and `DimensionResult` schema are
identical to the original RAGAS-backed design.

---

## Cost

Each chat turn makes up to 9 OpenAI API calls:
- 1× `text-embedding-3-small` (query embedding) — ~$0.00002
- 4× `gpt-4o` (RAGAS-style judge calls: faithfulness, relevancy, precision, recall) — ~$0.004
- 4× `gpt-4o` (DeepEval G-Eval calls: completeness, coherence, balance, toxicity) — ~$0.004

Roughly **$0.008 per turn**. A 20-turn demo session costs under twenty cents.

Lexical metrics (BLEU, ROUGE-L, BERTScore) run on the golden dataset only — zero API cost.
