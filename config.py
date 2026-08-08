"""Configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH: str = os.getenv("DOCS_PATH", "./docs")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Azure OpenAI
AZURE_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
RRF_K: int = int(os.getenv("RRF_K", "60"))
TOP_K: int = int(os.getenv("TOP_K", "6"))
RERANK_POOL: int = int(os.getenv("RERANK_POOL", "15"))

# Embeddings
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


def validate() -> None:
    """Raise if required Azure credentials are missing."""
    missing = []
    if not AZURE_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_KEY:
        missing.append("AZURE_OPENAI_KEY")
    if not AZURE_DEPLOYMENT:
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
