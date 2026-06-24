"""
config.py — Application settings loaded from the .env file.

Pydantic-settings reads environment variables automatically, so you never
need to call os.environ directly anywhere else in the codebase.
"""

from google.oauth2 import service_account
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required
    openai_api_key: str

    # Vertex AI — service account + project
    google_application_credentials: str = "./credentials/client_secrets.json"
    vertex_project_id: str = ""
    vertex_region: str = "us-east5"

    # Model choices
    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-sonnet-4-6"
    eval_model: str = "gpt-4o"          # judge model (different family from chatbot)
    eval_model_mini: str = "gpt-4o-mini"  # cheap classification calls
    embedding_model: str = "text-embedding-3-small"

    # RAG pipeline
    chroma_path: str = "./chroma_db"
    knowledge_path: str = "./knowledge"
    top_k_chunks: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 50
    collection_name: str = "knowledge_base"

    # Server ports
    api_port: int = 8000
    ui_port: int = 8501


def load_vertex_credentials() -> service_account.Credentials:
    return service_account.Credentials.from_service_account_file(
        settings.google_application_credentials,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


# Singleton — import this object everywhere instead of re-instantiating.
settings = Settings()
