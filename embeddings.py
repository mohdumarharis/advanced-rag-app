"""Embedding function factory."""
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma's HNSW metric. The MiniLM and BGE families are trained for cosine;
# with L2 (Chroma's default) on unnormalized vectors, ranking becomes
# magnitude-sensitive and no longer matches what the model was trained to do.
DISTANCE_SPACE = "cosine"


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Return the embedding function used for both indexing and querying."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
