"""Batch test script for the RAG pipeline."""
import config
from index import load_or_build_index
from retriever import HybridRRFRetriever
from rag_chain import invoke


def run_tests() -> None:
    config.validate()

    print("Checking index...")
    vectorstore, chunks = load_or_build_index(
        docs_path=config.DOCS_PATH,
        persist_dir=config.CHROMA_PERSIST_DIR,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"Index ready: {len(chunks)} chunks")

    retriever = HybridRRFRetriever(
        vectorstore=vectorstore,
        chunks=chunks,
        reranker_model=config.RERANKER_MODEL,
        k=config.RRF_K,
        top_k=config.TOP_K,
        rerank_pool=config.RERANK_POOL,
    )
    print("BM25 sparse index built")
    print("Reranker loaded")
    print()

    test_queries = [
        "What does RAM-UR stand for?",
        "What are the three axioms in Theorem 1?",
        "How does RAM-UR differ from Kovach and Suleymanov 2023?",
        "What is the functional form of RAM-UR-IRA in Definition 8?",
        "What dataset was used to test the model?",
        "Does the paper include Python code?",
    ]

    for q in test_queries:
        print("=" * 60)
        print(f"Q: {q}")
        answer, sources = invoke(q, retriever)
        print(f"A: {answer[:300]}...")
        print(f"Sources: {len(sources)} chunks retrieved")
        print()


if __name__ == "__main__":
    run_tests()
