# advanced-rag-app

A retrieval-augmented generation pipeline over local documents (PDF, PPTX, TXT, MD),
built on hybrid retrieval and Azure OpenAI.

The retrieval path is three stages rather than the usual single vector lookup:

```
documents ──► chunk ──► ┌─ dense  (Chroma + MiniLM, cosine) ─┐
                        │                                    ├─► RRF fusion ──► cross-encoder rerank ──► top-k
                        └─ sparse (BM25 over the same chunks)┘
```

Dense retrieval alone misses exact terms — acronyms, notation, proper nouns. BM25 alone
misses paraphrase. Reciprocal Rank Fusion combines both rankings, and a cross-encoder
reranks the pooled candidates by actually reading each one against the query.

## Setup

```bash
python -m venv .ragenv && source .ragenv/bin/activate
pip install -r requirements.txt
```

PPTX image OCR needs Tesseract on the system:

```bash
brew install tesseract   # or: apt install tesseract-ocr
```

Copy `.env.example` to `.env` and fill in your Azure OpenAI endpoint, key, and deployment.
`.env` is gitignored — keep it that way.

Then drop documents into `docs/`. Nothing is included in this repo; bring your own.

## Usage

```bash
python main.py
```

The first run parses, chunks, embeds, and writes a Chroma index to `chroma_db/`.
Subsequent runs hash every source file and re-embed **only what changed** — a manifest
in the index directory tracks which chunk IDs came from which file. Delete a document and
its rows are removed; edit one and only its rows are rebuilt.

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL` changes the index signature
and forces a full rebuild, so vectors from two different models can never mix.

## Evaluation

The reason this repo has an eval harness: without one you cannot tell whether changing
chunk size or swapping the embedding model helped or hurt.

```bash
python evaluate.py                                  # retrieval only — no Azure needed, free
python evaluate.py --generate                       # + answer checks (one LLM call per case)
python evaluate.py --save eval/results/baseline.json
python evaluate.py --compare eval/results/baseline.json
CHUNK_SIZE=600 python evaluate.py --compare eval/results/baseline.json
```

Retrieval scoring needs no credentials at all, so tuning chunking and embeddings costs
nothing. Every stage is scored separately — a single end-to-end number tells you something
broke, a per-stage breakdown tells you *where*:

```
stage        hit@k     mrr  recall
dense        0.500   0.420   0.222
sparse       0.800   0.667   0.526
fused        0.800   0.437   0.461
reranked     0.800   0.667   0.558   ← final
```

Ground truth in `eval/golden.json` is expressed as **text markers, not chunk IDs**, so the
golden set survives re-chunking — otherwise the set would break on exactly the changes you
want to measure. The harness exits non-zero if any case's markers stop matching the corpus,
so a rotted golden set can't pass silently.

`eval/golden.json` and the checked-in baseline are written against a specific
economics paper (arXiv:2407.01528) that is **not** distributed here. Replace the cases with
ones drawn from your own corpus.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Environment-driven settings and credential validation |
| `loaders.py` | Per-file loading: PyMuPDF, python-pptx (text/tables/image OCR), plain text |
| `chunker.py` | Recursive splitting with content-addressed, re-index-stable chunk IDs |
| `embeddings.py` | Embedding function (normalized, cosine) |
| `index.py` | Persistent index: file hashing, manifest, incremental add/delete |
| `retriever.py` | Dense + BM25 + RRF + cross-encoder rerank |
| `llm_client.py` | Lazily constructed Azure OpenAI client |
| `rag_chain.py` | Prompt assembly and invocation |
| `evaluate.py` | Evaluation harness |

## Known limitations

Honest list, roughly in priority order:

- **Dependencies are unpinned.** `langchain-community` + `chromadb` + `sentence-transformers`
  moves fast; this will break. Pin before relying on it.
- **`langchain_community.Chroma` is deprecated** and warns on every run — should move to
  the `langchain-chroma` package.
- **Dense retrieval underperforms BM25** on technical prose (0.222 vs 0.526 recall in the
  baseline above). A stronger embedding model than MiniLM-L6 is the obvious next experiment.
- **RRF is unweighted**, and on that corpus fusion scored below sparse alone. Weighted
  fusion is worth testing.
- **Reranker scores are discarded** — top-k is returned regardless of relevance, so nothing
  is filtered out when the corpus simply doesn't contain the answer.
- **No error handling in the CLI loop**; an API error ends the session.
- **Document discovery is non-recursive** — files in `docs/` subdirectories are skipped.
- **No conversation memory**, so follow-up questions aren't resolved against prior turns.
