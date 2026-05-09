# config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Central configuration for Project Aegis RAG."""

    # === API Keys ===
    OPENAI_API_KEY: str | None = None
    PINECONE_API_KEY: str = ""          # ← This is what embedder.py imports

    # === Pinecone Settings ===
    PINECONE_INDEX_NAME: str = "project-aegis-policies"
    PINECONE_ENVIRONMENT: str = "us-east-1"   # For serverless

    # === Embedding Model ===
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # === Chunking Settings ===
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP_PERCENT: float = 0.12

    # === Retrieval Settings ===
    TOP_K: int = 25
    RERANK_TOP_N: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()

# Optional: Quick validation
if not settings.PINECONE_API_KEY:
    print("⚠️  Warning: PINECONE_API_KEY is not set in .env file!")