# advanced-rag-app

[![Python application](https://github.com/mohdumarharis/advanced-rag-app/actions/workflows/python-app.yml/badge.svg)](https://github.com/mohdumarharis/advanced-rag-app/actions/workflows/python-app.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

**A RAG pipeline built around the two things most RAG projects skip: retrieval that
combines more than one method, and a way to prove whether any change actually helped.**

Ask questions about your own PDFs, slide decks, and notes. Answers are grounded in
retrieved passages — and the console shows you the retrieval that produced them, including
how confident the reranker was about every chunk it kept.

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

---

## Why this is different

The standard RAG walkthrough is four steps: load → chunk → embed → nearest-neighbour
lookup → stuff into a prompt. It demos well and degrades quietly, in three specific ways.

| | Typical RAG project | This repo |
| --- | --- | --- |
| **Retrieval** | One dense vector lookup | Dense + BM25, fused with RRF, then cross-encoder reranked |
| **Indexing** | Rebuilt from scratch on every run | Content-hashed manifest; only changed files are re-embedded |
| **Chunk identity** | Positional (`chunk_0`, `chunk_1`…) | Content-addressed SHA-256, stable across re-indexing |
| **Evaluation** | None — you eyeball a few answers | Golden set with per-stage recall, MRR, and abstention scoring |
| **Retrieval visibility** | Hidden; you see only the answer | Per-stage trace and reranker confidence, in the UI |
| **Tuning** | Guesswork | `--compare` against a saved baseline |

### 1 · Retrieval combines two methods that fail differently

```
documents ──► chunk ──► ┌─ dense  (Chroma + MiniLM, cosine) ─┐
                        │                                    ├─► RRF ──► cross-encoder ──► top-k
                        └─ sparse (BM25 over the same chunks)┘
```

Embeddings capture paraphrase but blur exact tokens — acronyms, notation, proper nouns,
identifiers. BM25 nails exact tokens and misses paraphrase entirely. Reciprocal Rank Fusion
merges the two rankings without needing their scores to be on a comparable scale, and a
cross-encoder then re-reads each pooled candidate *against the query* rather than comparing
two independently-computed vectors.

That last stage is the expensive one, and this repo measures whether it's worth it rather
than assuming so. On the reference corpus, it is — see below.

### 2 · The index is incremental, which required rethinking chunk IDs

Most implementations delete and rebuild the vector store on startup, which makes the
`persist_directory` decorative and re-runs OCR and embedding over documents that never
changed.

Here, a manifest stores the SHA-256 of every source file and the chunk IDs it produced.
On startup, only files whose hash changed are re-read; deleted files have their rows
removed; everything else is left alone.

That is only possible because **chunk IDs are content hashes rather than positions**. With
positional IDs, inserting a paragraph into the first document shifts every downstream ID,
so no row can be located and replaced later — which is precisely why the wipe-and-rebuild
pattern is so common. Content addressing removes the constraint.

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL` alters the index signature and
forces a full rebuild, so vectors from two different models can never end up mixed in one
collection.

### 3 · Every stage is scored separately

A single end-to-end score tells you something broke. A per-stage breakdown tells you
*where*, which is the difference between tuning and guessing.

```bash
python evaluate.py                                  # retrieval only — no API key needed, free
python evaluate.py --generate                       # + answer checks (one LLM call per case)
python evaluate.py --save eval/results/baseline.json
python evaluate.py --compare eval/results/baseline.json
CHUNK_SIZE=600 python evaluate.py --compare eval/results/baseline.json
```

Retrieval scoring needs no credentials at all, so sweeping chunk sizes or swapping
embedding models costs nothing.

Ground truth in `eval/golden.json` is expressed as **text markers, not chunk IDs**. Pinning
golden chunks by ID would break the moment you changed chunk size — exactly the change you
most want to measure. The harness exits non-zero if a case's markers stop matching anything
in the corpus, so a golden set that has rotted fails loudly instead of quietly scoring zero.

---

## What the measurements actually show

From the reference corpus (a 26-page economics paper, 70 chunks, 13 golden cases):

```
stage        hit@6     mrr  recall
dense        0.500   0.420   0.222
sparse       0.800   0.667   0.526
fused        0.800   0.437   0.461
reranked     0.800   0.667   0.558   ← final

answerable answered correctly   0.900
unanswerable declined correctly 1.000
```

Read honestly, that table contains three findings — two of them unflattering, which is the
point of having measurements at all:

- **Dense retrieval is the weakest component here**, not the strongest. BM25 recall is more
  than double it. On notation-heavy prose full of acronyms, a small general-purpose
  embedding model contributes less than a keyword index. A hybrid pipeline is what keeps
  this from being fatal.
- **Unweighted RRF currently scores below sparse alone** (0.461 vs 0.526) — equal-weight
  fusion dilutes strong BM25 hits with weak dense ones. Weighted fusion is the obvious next
  experiment, and it is now a measurable one rather than an opinion.
- **The cross-encoder earns its cost**, recovering that dilution and finishing highest at
  0.558. That is a measured claim, not an assumption.

Abstention is the quiet win: all three unanswerable questions — including one entirely
outside the corpus — returned "I don't know" instead of inventing an answer, and that
behaviour now has a regression test.

---

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

Then drop documents into `docs/`. Nothing is bundled; bring your own.

## Usage

```bash
python main.py
```

| Command | |
| --- | --- |
| `/sources` | full text of the chunks behind the last answer |
| `/trace` | candidate counts for each retrieval stage |
| `/config` | active models and retrieval settings |
| `/help` | command list |
| `/quit` | exit (Ctrl-D and Ctrl-C also work) |

Ctrl-C during a query abandons that question without tearing down the loaded index, and a
failed API call is reported in place rather than ending the session.

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

- **Dense retrieval underperforms BM25 on technical prose** (0.222 vs 0.526 recall above).
  A stronger embedding model than MiniLM-L6 is the highest-value next change.
- **RRF is unweighted**, and scored below sparse alone on the reference corpus.
- **Reranker scores are shown but not acted on** — top-k is returned regardless of
  relevance, so nothing is filtered when the corpus simply lacks the answer. The console
  displays the scores, so you can see when it happens.
- **Document discovery is non-recursive** — files in `docs/` subdirectories are skipped.
- **PDF line-break hyphenation isn't repaired** (`fail-\ning`), costing BM25 a match on any
  word split across a line.
- **No conversation memory**, so follow-up questions aren't resolved against prior turns.
- **`langchain_community.Chroma` is deprecated** and warns on every run; it should move to
  the `langchain-chroma` package.
- **CI runs lint only.** It does not import the package, so a missing dependency can pass
  green — that has happened once already in this repo's history.

The golden set and checked-in baseline are written against a specific paper
(arXiv:2407.01528) that is **not** redistributed here. Replace the cases with ones drawn
from your own corpus.
