"""CLI entry point for the RAG application."""
import config
from index import load_or_build_index
from retriever import HybridRRFRetriever
from rag_chain import invoke


def main() -> None:
    config.validate()

    # 1. Sync the persistent index (only new/changed files are re-embedded)
    print("Checking index...")
    vectorstore, chunks = load_or_build_index(
        docs_path=config.DOCS_PATH,
        persist_dir=config.CHROMA_PERSIST_DIR,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"Index ready: {len(chunks)} chunks")

    # 2. Hybrid retriever
    retriever = HybridRRFRetriever(
        vectorstore=vectorstore,
        chunks=chunks,
        reranker_model=config.RERANKER_MODEL,
        k=config.RRF_K,
        top_k=config.TOP_K,
        rerank_pool=config.RERANK_POOL,
    )
    print("Hybrid retriever ready")

    # 3. Interactive loop
    while True:
        query = input("\nAsk a question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        answer, sources = invoke(query, retriever)
        print("\n--- Answer ---")
        print(answer)

        print("\n--- Sources (Hybrid + Reranker) ---")
        for i, doc in enumerate(sources, 1):
            src = doc.metadata.get("source", "unknown")
            slide = doc.metadata.get("slide", "N/A")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"{i}. {src} (slide {slide}) | {preview}...")


if __name__ == "__main__":
    main()
