"""
routers/chat.py — POST /api/chat

Core endpoint that:
  1. Receives the user's question and optional domain_filter
  2. Runs the RAG pipeline (embed → retrieve → augment)
  3. Generates an answer using Gemini via Vertex AI
  4. Runs all eight eval metrics concurrently via run_evals()
  5. Persists the result to SQLite
  6. Returns the answer, retrieved chunks, eval scores, and latency

OpenAI is used only for embeddings. Generation is handled by Gemini.
"""

import json
import time
import uuid

from fastapi import APIRouter
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import load_vertex_credentials, settings
from database import save_conversation
from eval.models import EvalResult
from eval.runner import run_evals
from rag.retriever import retrieve_chunks

router = APIRouter()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    question: str
    domain_filter: str | None = None


class ChatResponse(BaseModel):
    message_id: str
    question: str
    answer: str
    chunks: list[dict]
    eval_result: EvalResult
    latency_ms: float


# --- System prompt ---
# The model is told to answer ONLY from the provided context. This makes
# faithfulness evaluation more meaningful — the model should stay grounded.
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions about world history. "
    "You must base your answers ONLY on the context provided to you. "
    "If the context does not contain enough information to answer the question, "
    "say so clearly rather than guessing. "
    "Be concise and accurate. Cite specific details from the context where possible."
)


def _get_gemini_client() -> genai.Client:
    """Create a Gemini client authenticated via Vertex AI service account."""
    return genai.Client(
        vertexai=True,
        project=settings.vertex_project_id,
        location=settings.vertex_region,
        credentials=load_vertex_credentials(),
    )


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Assemble the augmented prompt with retrieved context."""
    context_text = "\n\n---\n\n".join(
        f"[Source {i + 1}: {chunk['source']}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above:"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle a chat turn: RAG pipeline + Gemini generation + eval + persist + return.
    """
    # Record start time before anything happens — this is the latency baseline
    start_time = time.perf_counter()

    # OpenAI client is used only for embeddings
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Step 1: Retrieve relevant chunks from ChromaDB
    chunks = await retrieve_chunks(
        query=request.question,
        client=openai_client,
        domain_filter=request.domain_filter,
    )

    # Step 2: Build augmented prompt and generate answer via Gemini
    prompt = _build_prompt(request.question, chunks)

    gemini = _get_gemini_client()
    resp = gemini.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    answer: str = resp.text or ""

    # Step 3: Run all eight eval metrics concurrently
    context_texts = [chunk["text"] for chunk in chunks]
    eval_result = await run_evals(
        question=request.question,
        context=context_texts,
        answer=answer,
        reference_answer=None,
        start_time=start_time,
        openai_client=openai_client,
    )

    # Step 4: Determine domain tag from first chunk (fall back to "general")
    domain_tag = chunks[0]["domain_tag"] if chunks else "general"

    # Step 5: Persist to SQLite
    message_id = str(uuid.uuid4())
    save_conversation({
        "id": message_id,
        "question": request.question,
        "answer": answer,
        # Legacy columns — kept for backward compatibility
        "retrieved_chunks": context_texts,
        "chunk_sources": [c["source"] for c in chunks],
        "faithfulness": None,
        "faith_reason": eval_result.faithfulness.reason,
        "faith_passed": int(eval_result.faithfulness.passed),
        "relevancy": None,
        "relevancy_passed": int(eval_result.answer_relevancy.passed),
        "precision": None,
        "precision_passed": int(eval_result.context_precision.passed),
        "latency_ms": eval_result.latency_ms,
        "overall_passed": int(eval_result.overall_passed),
        # New 8-dimension columns
        "domain_tag": domain_tag,
        "answer_relevancy_passed": int(eval_result.answer_relevancy.passed),
        "completeness_passed": int(eval_result.completeness.passed),
        "context_recall_passed": int(eval_result.context_recall.passed),
        "coherence_passed": int(eval_result.coherence.passed),
        "historical_balance_passed": int(eval_result.historical_balance.passed),
        "toxicity_passed": int(eval_result.toxicity.passed),
        "checklist_json": eval_result.model_dump(),
    })

    return ChatResponse(
        message_id=message_id,
        question=request.question,
        answer=answer,
        chunks=chunks,
        eval_result=eval_result,
        latency_ms=eval_result.latency_ms,
    )
