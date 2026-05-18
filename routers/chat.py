"""
routers/chat.py — POST /api/chat

This is the core endpoint. It:
  1. Receives the user's question
  2. Runs the RAG pipeline (embed → retrieve → augment → generate)
  3. Runs all four eval metrics concurrently
  4. Persists the result to SQLite
  5. Returns the answer, retrieved chunks, and eval scores in one response

The OpenAI client is created once per request lifecycle using FastAPI's
dependency injection. For production use you would create it once at startup
and inject it as an app state dependency instead.
"""

import time
import uuid

from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import settings
from database import save_conversation
from eval.runner import run_evals
from eval.models import EvalResult
from rag.retriever import retrieve_chunks

router = APIRouter()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    message_id: str
    question: str
    answer: str
    retrieved_chunks: list[str]
    chunk_sources: list[str]
    eval_result: EvalResult


# --- System prompt ---
# The model is told to answer ONLY from the provided context. This makes
# faithfulness evaluation more meaningful — the model should stay grounded.
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about
machine learning and AI concepts. You must base your answers ONLY on the
context provided to you. If the context does not contain enough information
to answer the question, say so clearly rather than guessing.

Be concise and accurate. Cite specific details from the context where possible."""


def build_user_prompt(question: str, context_chunks: list[str]) -> str:
    """Assemble the augmented prompt with retrieved context."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )
    return f"""Context:
{context_text}

Question: {question}

Answer based only on the context above:"""


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle a chat turn: RAG pipeline + eval + persist + return.
    """
    # Record start time before anything happens — this is the latency baseline
    start_time = time.perf_counter()

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Step 1: Retrieve relevant chunks from ChromaDB
    chunks, sources = await retrieve_chunks(
        query=request.question,
        client=client,
    )

    # Step 2: Build the augmented prompt and call the chat model
    user_prompt = build_user_prompt(request.question, chunks)

    completion = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Low temperature for factual consistency
    )
    answer = completion.choices[0].message.content or ""

    # Step 3: Run all eval metrics concurrently
    eval_result = await run_evals(
        question=request.question,
        context=chunks,
        answer=answer,
        start_time=start_time,
        client=client,
    )

    # Step 4: Persist to SQLite
    message_id = str(uuid.uuid4())
    save_conversation({
        "id": message_id,
        "question": request.question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "chunk_sources": sources,
        "faithfulness": eval_result.faithfulness.score,
        "faith_reason": eval_result.faithfulness.reason,
        "faith_passed": int(eval_result.faithfulness.passed),
        "relevancy": eval_result.answer_relevancy.score,
        "relevancy_passed": int(eval_result.answer_relevancy.passed),
        "precision": eval_result.context_precision.score,
        "precision_passed": int(eval_result.context_precision.passed),
        "latency_ms": eval_result.latency_ms,
        "overall_passed": int(eval_result.overall_passed),
    })

    return ChatResponse(
        message_id=message_id,
        question=request.question,
        answer=answer,
        retrieved_chunks=chunks,
        chunk_sources=sources,
        eval_result=eval_result,
    )
