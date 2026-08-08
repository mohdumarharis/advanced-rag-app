# advanced-rag-app

**Hybrid-retrieval RAG over your own documents — dense + BM25, reciprocal rank fusion,
and cross-encoder reranking, with an eval harness that makes every change measurable.**

Ask questions about local PDFs, slide decks, and notes. Answers are grounded in retrieved
passages, and the console shows you the retrieval that produced them — which stage found
what, and how confident the reranker was about each chunk it kept.

```
❯ Which three axioms characterize the RAM-UR?

╭──────────────────────────────────────────────────────────────────────╮
│  The three axioms that characterize the RAM-UR are:                  │
│                                                                      │
│   1 Existence of a Dominant Alternative (EDA)                        │
│   2 Certainty WARP (C-WARP)                                          │
│   3 Expansion (EXP)                                                  │
╰──────────────────────────────────────────────────────────────────────╯

  #   relevance        source                loc     excerpt
 ──────────────────────────────────────────────────────────────────────
  1   ━━━━━━╌╌ 0.73    paper.pdf             p.6     3 Characterization In the…
  2   ━━━━━━╌╌ 0.73    paper.pdf             p.9     all alternatives, the RAM…
  3   ━━━━━━╌╌ 0.72    paper.pdf             p.3     mean that this product wi…

                     dense 15  ·  sparse 15  ·  union 23  →  answered from 6
```

Most RAG front-ends show only an answer, which makes a bad answer impossible to diagnose.
This one keeps the pipeline visible.

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

Inside the console:

| Command | |
| --- | --- |
| `/sources` | full text of the chunks behind the last answer |
| `/trace` | candidate counts for each retrieval stage |
| `/config` | active models and retrieval settings |
| `/help` | command list |
| `/quit` | exit (Ctrl-D and Ctrl-C also work) |

Ctrl-C during a query abandons that question without tearing down the loaded index, and a
failed API call is reported in place rather than ending the session.

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
| `main.py` | Interactive console |
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
- **Reranker scores are shown but not acted on** — top-k is returned regardless of
  relevance, so nothing is filtered out when the corpus simply doesn't contain the answer.
  The console displays the scores, so you can see when this happens.
- **Document discovery is non-recursive** — files in `docs/` subdirectories are skipped.
- **PDF line-break hyphenation isn't repaired** (`fail-\ning`), which costs BM25 a match
  on any word split across a line.
- **No conversation memory**, so follow-up questions aren't resolved against prior turns.
