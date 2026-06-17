"""
main.py — FastAPI application entry point.

The lifespan context manager runs two startup tasks:
  1. init_db()  — creates the SQLite table if it doesn't exist
  2. index_knowledge_base()  — indexes .txt files into ChromaDB (skipped if
     the collection already contains documents)

Streamlit UI runs as its own process on port 8501 — FastAPI no longer
mounts a static folder. The two processes communicate via HTTP.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from rag.indexer import index_knowledge_base
from routers import chat, history
from routers import eval_runs as eval_runs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    # --- Startup ---
    print("Initialising database...")
    init_db()
    print("Database ready.")

    print(f"Checking knowledge base at '{settings.knowledge_path}'...")
    await index_knowledge_base()
    print("Knowledge base ready.")

    yield  # Application runs here

    # --- Shutdown (nothing to clean up for this demo) ---
    print("Shutting down.")


app = FastAPI(
    title="RAG Eval Chatbot",
    description="A chatbot that makes AI evaluations visible.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow the Streamlit frontend (port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{settings.ui_port}",
        f"http://127.0.0.1:{settings.ui_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(chat.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(eval_runs_router.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    """Liveness check endpoint."""
    return {"status": "ok", "service": "rag-chatbot-eval"}
