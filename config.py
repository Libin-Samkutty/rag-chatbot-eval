"""
config.py — Application settings loaded from the .env file.

Pydantic-settings reads environment variables automatically, so you never
need to call os.environ directly anywhere else in the codebase.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # The only required value — everything else has a default.
    openai_api_key: str

    chroma_path: str = "./chroma_db"
    knowledge_path: str = "./knowledge"
    top_k_chunks: int = 4
    chat_model: str = "gpt-4o-mini"
    eval_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "knowledge_base"

    class Config:
        env_file = ".env"
        # Allow extra variables in .env without raising an error
        extra = "ignore"


# Singleton — import this object everywhere instead of re-instantiating.
settings = Settings()
