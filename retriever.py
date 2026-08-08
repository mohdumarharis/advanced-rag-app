"""Hybrid retriever: dense (Chroma) + sparse (BM25) + RRF + cross-encoder rerank."""
from typing import Dict, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRRFRetriever:
    """
    Reciprocal Rank Fusion of dense and sparse results,
    followed by a cross-encoder reranker.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: List[Document],
        reranker_model: str = "BAAI/bge-reranker-base",
        k: int = 60,
        top_k: int = 6,
        rerank_pool: int = 15,
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k = k
        self.top_k = top_k
        self.rerank_pool = rerank_pool

        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.reranker = CrossEncoder(reranker_model)

    @staticmethod
    def _key(doc: Document) -> str:
        """Fusion key. Falls back to content so a missing chunk_id cannot KeyError."""
        return doc.metadata.get("chunk_id") or doc.page_content

    def _dense(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.rerank_pool)

    def _sparse(self, query: str) -> List[Document]:
        bm25_scores = self.bm25.get_scores(query.lower().split())
        indexed = list(enumerate(bm25_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, _ in indexed[:self.rerank_pool]]

    def _fuse(self, dense: List[Document], sparse: List[Document]) -> List[Document]:
        """Reciprocal Rank Fusion over the two candidate lists."""
        scores: dict = {}
        doc_map: dict = {}

        for rank, doc in enumerate(dense):
            cid = self._key(doc)
            scores[cid] = scores.get(cid, 0) + 1 / (self.k + rank + 1)
            doc_map[cid] = doc

        for rank, doc in enumerate(sparse):
            cid = self._key(doc)
            scores[cid] = scores.get(cid, 0) + 1 / (self.k + rank + 1)
            if cid not in doc_map:
                doc_map[cid] = doc

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[cid] for cid, _ in ordered][:self.rerank_pool]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scored = list(zip(docs, self.reranker.predict(pairs)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored]

    def retrieve_stages(self, query: str) -> Dict[str, List[Document]]:
        """
        Every intermediate stage, untruncated.

        The eval harness scores each stage separately so you can see what
        fusion and reranking actually contribute over dense retrieval alone.
        """
        dense = self._dense(query)
        sparse = self._sparse(query)
        fused = self._fuse(dense, sparse)
        return {
            "dense": dense,
            "sparse": sparse,
            "fused": fused,
            "reranked": self._rerank(query, fused),
        }

    def retrieve(self, query: str) -> List[Document]:
        return self.retrieve_stages(query)["reranked"][:self.top_k]
