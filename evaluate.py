"""
Offline evaluation harness for the RAG pipeline.

Two independent layers:

  Retrieval  Deterministic, free, and needs no Azure credentials. Scores every
             pipeline stage separately (dense / bm25 / fused / reranked) so you
             can see what fusion and the cross-encoder actually contribute.

  Generation Opt-in via --generate. Sends one LLM call per case, so it costs
             money. Checks answer markers and, for unanswerable questions,
             whether the model correctly declines instead of inventing.

Usage
    python evaluate.py                              # retrieval only
    python evaluate.py --save eval/results/base.json
    python evaluate.py --compare eval/results/base.json
    python evaluate.py --generate                   # + LLM checks
    python evaluate.py --k 10
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Set

import config
from index import load_or_build_index
from retriever import HybridRRFRetriever

GOLDEN_PATH = os.path.join("eval", "golden.json")
STAGES = ("dense", "sparse", "fused", "reranked")


# --------------------------------------------------------------------------
# text matching
# --------------------------------------------------------------------------

def norm(text: str) -> str:
    """
    Normalize for marker matching.

    NFKC folds the PDF's 'ﬁ'/'ﬀ' ligatures into plain ASCII, which otherwise
    make perfectly good markers silently fail to match extracted text.
    """
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_all(haystack: str, markers: Sequence[str]) -> bool:
    hay = norm(haystack)
    return all(norm(m) in hay for m in markers)


def contains_any(haystack: str, markers: Sequence[str]) -> bool:
    hay = norm(haystack)
    return any(norm(m) in hay for m in markers)


# --------------------------------------------------------------------------
# retrieval scoring
# --------------------------------------------------------------------------

def relevant_ids(chunks, markers: Sequence[str]) -> Set[str]:
    """Every chunk in the corpus that satisfies the case's markers."""
    return {
        c.metadata["chunk_id"]
        for c in chunks
        if contains_all(c.page_content, markers)
    }


def score_stage(docs, gold: Set[str], k: int) -> Dict[str, float]:
    top = docs[:k]
    ids = [d.metadata.get("chunk_id") for d in top]
    hits = [i for i, cid in enumerate(ids) if cid in gold]
    return {
        "hit": 1.0 if hits else 0.0,
        "mrr": 1.0 / (hits[0] + 1) if hits else 0.0,
        "recall": len(set(ids) & gold) / len(gold) if gold else 0.0,
    }


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run(cases, chunks, retriever, k: int, generate: bool) -> Dict:
    answerable = [c for c in cases if not c.get("unanswerable")]
    unanswerable = [c for c in cases if c.get("unanswerable")]

    broken: List[str] = []
    per_case: List[Dict] = []

    for case in answerable:
        gold = relevant_ids(chunks, case["relevant_contains"])
        if not gold:
            # The markers match nothing in the corpus, so this case can only
            # ever score zero. That is a bug in the golden set, not the app.
            broken.append(case["id"])
            continue

        stages = retriever.retrieve_stages(case["question"])
        record = {
            "id": case["id"],
            "gold_chunks": len(gold),
            "stages": {s: score_stage(stages[s], gold, k) for s in STAGES},
        }
        if generate:
            record["generation"] = check_generation(case, stages["reranked"][:k])
        per_case.append(record)

    gen_unans: List[Dict] = []
    if generate:
        for case in unanswerable:
            docs = retriever.retrieve(case["question"])
            gen_unans.append({"id": case["id"], **check_generation(case, docs)})

    summary = {
        "k": k,
        "chunks_indexed": len(chunks),
        "cases_scored": len(per_case),
        "broken_cases": broken,
        "retrieval": {
            stage: {
                metric: mean([c["stages"][stage][metric] for c in per_case])
                for metric in ("hit", "mrr", "recall")
            }
            for stage in STAGES
        },
        "per_case": per_case,
    }
    if generate:
        answered_ok = [c["generation"]["pass"] for c in per_case]
        declined_ok = [c["pass"] for c in gen_unans]
        summary["generation"] = {
            "answerable_correct": mean(answered_ok),
            "abstention_correct": mean(declined_ok),
            "unanswerable_cases": gen_unans,
        }
    return summary


def check_generation(case, docs) -> Dict:
    from rag_chain import invoke  # imported lazily: needs Azure credentials

    class _Fixed:
        """Feed the already-retrieved docs to the chain, so generation is
        scored on retrieval we have already measured rather than a second,
        differently-seeded retrieval pass."""
        def retrieve(self, _query):
            return docs

    answer, _ = invoke(case["question"], _Fixed())
    markers = case["answer_contains"]
    matcher = contains_any if case.get("answer_match") == "any" else contains_all
    return {"pass": matcher(answer, markers), "answer": answer.strip()[:240]}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def bar(value: float, width: int = 18) -> str:
    filled = round(value * width)
    return "█" * filled + "·" * (width - filled)


def report(summary: Dict, baseline: Optional[Dict]) -> None:
    k = summary["k"]
    print()
    print(f"RETRIEVAL  k={k}  ·  {summary['cases_scored']} answerable cases  "
          f"·  {summary['chunks_indexed']} chunks indexed")
    print("-" * 72)
    print(f"{'stage':<10} {'hit@k':>7} {'mrr':>7} {'recall':>7}   {'recall@k':<18}")

    base_r = (baseline or {}).get("retrieval", {})
    for stage in STAGES:
        row = summary["retrieval"][stage]
        delta = ""
        if stage in base_r:
            d = row["recall"] - base_r[stage]["recall"]
            if abs(d) >= 0.001:
                delta = f"  {d:+.3f}"
        print(f"{stage:<10} {row['hit']:>7.3f} {row['mrr']:>7.3f} "
              f"{row['recall']:>7.3f}   {bar(row['recall'])}{delta}")

    misses = [c for c in summary["per_case"] if c["stages"]["reranked"]["hit"] == 0.0]
    if misses:
        print()
        print("MISSED after reranking:")
        for c in misses:
            print(f"  · {c['id']:<24} "
                  f"(fused hit={c['stages']['fused']['hit']:.0f} — "
                  f"{'reranker dropped it' if c['stages']['fused']['hit'] else 'never retrieved'})")

    if summary["broken_cases"]:
        print()
        print("BROKEN GOLDEN CASES — markers match no chunk in the corpus:")
        for cid in summary["broken_cases"]:
            print(f"  · {cid}")

    gen = summary.get("generation")
    if gen:
        print()
        print("GENERATION")
        print("-" * 72)
        print(f"  answerable answered correctly   {gen['answerable_correct']:.3f}")
        print(f"  unanswerable declined correctly {gen['abstention_correct']:.3f}")
        bad = [c for c in gen["unanswerable_cases"] if not c["pass"]]
        for c in bad:
            print(f"    ! {c['id']} did not decline: {c['answer'][:120]}")
    print()


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=GOLDEN_PATH)
    ap.add_argument("--k", type=int, default=None, help="cutoff (default: config.TOP_K)")
    ap.add_argument("--generate", action="store_true",
                    help="also score generated answers (one LLM call per case)")
    ap.add_argument("--save", metavar="PATH", help="write the report as JSON")
    ap.add_argument("--compare", metavar="PATH", help="show deltas against a saved report")
    args = ap.parse_args()

    k = args.k or config.TOP_K

    with open(args.golden, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]

    if args.generate:
        config.validate()  # fail before spending time indexing

    store, chunks = load_or_build_index(
        docs_path=config.DOCS_PATH,
        persist_dir=config.CHROMA_PERSIST_DIR,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    retriever = HybridRRFRetriever(
        vectorstore=store,
        chunks=chunks,
        reranker_model=config.RERANKER_MODEL,
        k=config.RRF_K,
        top_k=config.TOP_K,
        rerank_pool=config.RERANK_POOL,
    )

    started = time.time()
    summary = run(cases, chunks, retriever, k, args.generate)
    summary["elapsed_s"] = round(time.time() - started, 1)
    summary["settings"] = {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "rrf_k": config.RRF_K,
        "rerank_pool": config.RERANK_POOL,
    }

    baseline = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            baseline = json.load(fh)

    report(summary, baseline)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"saved → {args.save}\n")

    # Non-zero exit if any golden case is unscoreable, so CI catches a rotted set.
    sys.exit(1 if summary["broken_cases"] else 0)


if __name__ == "__main__":
    main()
