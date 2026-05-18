"""
main.py — FastAPI application entry point.

The lifespan context manager runs two startup tasks:
  1. init_db()  — creates the SQLite table if it doesn't exist
  2. index_knowledge_base()  — indexes .txt files into ChromaDB (skipped if
     the collection already contains documents)

The frontend (static/index.html) is served as a static file at the root so
users just open http://localhost:8000.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from rag.indexer import index_knowledge_base
from routers import chat, history


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    # --- Startup ---
    print("⚙️  Initialising database...")
    init_db()
    print("✅  Database ready.")

    print(f"📚  Checking knowledge base at '{settings.knowledge_path}'...")
    await index_knowledge_base()
    print("✅  Knowledge base ready.")

    yield  # Application runs here

    # --- Shutdown (nothing to clean up for this demo) ---
    print("👋  Shutting down.")


app = FastAPI(
    title="RAG Eval Chatbot",
    description="A chatbot that makes AI evaluations visible.",
    version="1.0.0",
    lifespan=lifespan,
)

# API routes
app.include_router(chat.router, prefix="/api")
app.include_router(history.router, prefix="/api")

# Serve the React frontend from the static/ directory.
# This must come AFTER the API routes so /api/... is not caught by StaticFiles.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
