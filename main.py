"""
Interactive console for the RAG pipeline.

Most RAG front-ends hide retrieval and show only an answer, which makes a bad
answer impossible to diagnose. This one keeps the three-stage pipeline visible:
you see each stage run, how many candidates it produced, and what the
cross-encoder actually thought of the chunks it returned.
"""
import math
import os
from typing import Dict, List, Optional

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import config
from index import load_or_build_index
from rag_chain import answer_from_docs
from retriever import HybridRRFRetriever, Scored

console = Console()

ACCENT = "bright_cyan"
MUTED = "grey50"
FAINT = "grey35"

STAGE_NOTE = {
    "dense": "embedding query · searching vectors",
    "sparse": "scoring BM25 across the corpus",
    "fused": "fusing rankings · reciprocal rank fusion",
    "reranked": "cross-encoder reading each candidate",
}

HELP = [
    ("/sources", "full text of the chunks behind the last answer"),
    ("/trace", "candidate counts for each retrieval stage"),
    ("/config", "active models and retrieval settings"),
    ("/help", "this list"),
    ("/quit", "exit  (Ctrl-D also works)"),
]


# ---------------------------------------------------------------- formatting

def confidence(logit: float) -> float:
    """bge-reranker emits logits; a sigmoid puts them on a readable 0–1 scale."""
    return 1.0 / (1.0 + math.exp(-logit))


def score_bar(value: float, width: int = 8) -> Text:
    filled = round(value * width)
    hue = "green" if value >= 0.66 else "yellow" if value >= 0.33 else "red"
    return Text("━" * filled + "╌" * (width - filled), style=hue)


def locate(meta: Dict) -> str:
    """Where a chunk came from inside its document."""
    if meta.get("slide") is not None:
        return f"slide {meta['slide']}"
    if meta.get("page") is not None:
        return f"p.{int(meta['page']) + 1}"
    return "—"


def excerpt(text: str, width: int = 58) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


# ---------------------------------------------------------------- rendering

def masthead(chunks: List, sources: int) -> Panel:
    title = Text("ADVANCED RAG", style=f"bold {ACCENT}")
    flow = Text("dense → sparse → RRF → cross-encoder", style=FAINT)

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style=MUTED, justify="right")
    facts.add_column()
    facts.add_row("index", f"{len(chunks)} chunks from {sources} document"
                           f"{'s' if sources != 1 else ''}")
    facts.add_row("embed", f"{config.EMBEDDING_MODEL.split('/')[-1]}  "
                           f"[{FAINT}]cosine, normalized[/]")
    facts.add_row("rerank", f"{config.RERANKER_MODEL.split('/')[-1]}  "
                            f"[{FAINT}]pool {config.RERANK_POOL} → top "
                            f"{config.TOP_K}[/]")

    return Panel(
        Group(title, flow, Text(), facts),
        box=box.ROUNDED, border_style=ACCENT, padding=(1, 3),
    )


def sources_table(scored: List[Scored]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD, header_style=f"bold {MUTED}",
        padding=(0, 1), expand=True,
    )
    # Every column but the excerpt is fixed-width, so the excerpt is the only
    # thing that gives when the terminal is narrow.
    table.add_column("#", style=FAINT, width=2, justify="right")
    table.add_column("relevance", width=14, no_wrap=True)
    table.add_column("source", style=ACCENT, width=19, no_wrap=True,
                     overflow="ellipsis")
    table.add_column("loc", style=MUTED, width=5, no_wrap=True)
    table.add_column("excerpt", style=MUTED, ratio=1, no_wrap=True,
                     overflow="ellipsis")

    for rank, (doc, logit) in enumerate(scored, 1):
        value = confidence(logit)
        meter = Text.assemble(score_bar(value), " ", (f"{value:.2f}", FAINT))
        table.add_row(
            str(rank), meter,
            os.path.basename(str(doc.metadata.get("source", "?"))),
            locate(doc.metadata),
            excerpt(doc.page_content),
        )
    return table


def trace_line(counts: Dict[str, int]) -> Text:
    return Text.assemble(
        ("dense ", FAINT), (str(counts["dense"]), MUTED), ("  ·  ", FAINT),
        ("sparse ", FAINT), (str(counts["sparse"]), MUTED), ("  ·  ", FAINT),
        ("union ", FAINT), (str(counts["fused"]), MUTED), ("  →  ", FAINT),
        ("answered from ", FAINT), (str(counts["reranked"]), ACCENT),
    )


def show_help() -> None:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=ACCENT, no_wrap=True)
    grid.add_column(style=MUTED)
    for name, note in HELP:
        grid.add_row(name, note)
    console.print(Panel(grid, box=box.ROUNDED, border_style=FAINT,
                        title="commands", title_align="left", padding=(1, 2)))


def show_config() -> None:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=MUTED, justify="right")
    grid.add_column()
    for label, value in [
        ("docs", config.DOCS_PATH),
        ("index", config.CHROMA_PERSIST_DIR),
        ("chunk", f"{config.CHUNK_SIZE} chars, {config.CHUNK_OVERLAP} overlap"),
        ("embed", config.EMBEDDING_MODEL),
        ("rerank", config.RERANKER_MODEL),
        ("rrf k", str(config.RRF_K)),
        ("pool", f"{config.RERANK_POOL} → top {config.TOP_K}"),
        ("deployment", config.AZURE_DEPLOYMENT),
        ("max tokens", str(config.MAX_TOKENS)),
    ]:
        grid.add_row(label, str(value))
    console.print(Panel(grid, box=box.ROUNDED, border_style=FAINT,
                        title="config", title_align="left", padding=(1, 2)))


def show_full_sources(scored: List[Scored]) -> None:
    if not scored:
        console.print(f"[{MUTED}]Nothing retrieved yet.[/]")
        return
    for rank, (doc, logit) in enumerate(scored, 1):
        head = (f"[{ACCENT}]{rank}[/]  "
                f"{os.path.basename(str(doc.metadata.get('source', '?')))}  "
                f"[{MUTED}]{locate(doc.metadata)} · "
                f"score {confidence(logit):.2f}[/]")
        console.print(Panel(doc.page_content.strip(), title=head,
                            title_align="left", box=box.ROUNDED,
                            border_style=FAINT, padding=(1, 2)))


# ---------------------------------------------------------------- main loop

def ask(question: str, retriever: HybridRRFRetriever):
    """Retrieve, generate, and render. Returns (scored_docs, stage_counts)."""
    with console.status("", spinner="dots") as status:
        def on_stage(stage: str) -> None:
            status.update(f"[{ACCENT}]{STAGE_NOTE[stage]}[/]")

        scored, counts = retriever.retrieve_detailed(question, on_stage=on_stage)

        status.update(f"[{ACCENT}]composing an answer[/]")
        text = answer_from_docs(question, [doc for doc, _ in scored])

    console.print()
    console.print(Panel(Markdown(text.strip()), box=box.ROUNDED,
                        border_style=ACCENT, padding=(1, 2)))
    console.print(sources_table(scored))
    console.print(trace_line(counts), justify="right")
    return scored, counts


def main() -> None:
    config.validate()

    with console.status(f"[{ACCENT}]checking index[/]", spinner="dots"):
        vectorstore, chunks = load_or_build_index(
            docs_path=config.DOCS_PATH,
            persist_dir=config.CHROMA_PERSIST_DIR,
            embedding_model=config.EMBEDDING_MODEL,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    if not chunks:
        console.print(Panel(
            f"No documents indexed. Put PDF, PPTX, TXT, or MD files in "
            f"[{ACCENT}]{config.DOCS_PATH}[/] and start again.",
            box=box.ROUNDED, border_style="yellow", padding=(1, 2)))
        return

    with console.status(f"[{ACCENT}]loading reranker[/]", spinner="dots"):
        retriever = HybridRRFRetriever(
            vectorstore=vectorstore,
            chunks=chunks,
            reranker_model=config.RERANKER_MODEL,
            k=config.RRF_K,
            top_k=config.TOP_K,
            rerank_pool=config.RERANK_POOL,
        )

    documents = {c.metadata.get("source") for c in chunks}
    console.print()
    console.print(masthead(chunks, len(documents)))
    console.print(f"[{FAINT}]Ask a question, or /help for commands.[/]\n")

    last_scored: List[Scored] = []
    last_counts: Dict[str, int] = {}

    while True:
        try:
            question = console.input(f"[bold {ACCENT}]❯[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{MUTED}]bye[/]\n")
            return

        if not question:
            continue

        if question.startswith("/"):
            command = question.lower().split()[0]
            if command in ("/quit", "/exit", "/q"):
                console.print(f"[{MUTED}]bye[/]\n")
                return
            if command == "/help":
                show_help()
            elif command == "/config":
                show_config()
            elif command == "/sources":
                show_full_sources(last_scored)
            elif command == "/trace":
                console.print(trace_line(last_counts) if last_counts
                              else f"[{MUTED}]Nothing retrieved yet.[/]")
            else:
                console.print(f"[{MUTED}]Unknown command. /help for the list.[/]")
            console.print()
            continue

        try:
            last_scored, last_counts = ask(question, retriever)
        except KeyboardInterrupt:
            # Abandon this question, keep the session and the loaded index.
            console.print(f"\n[{MUTED}]cancelled[/]")
        except Exception as error:
            console.print(Panel(
                f"[bold red]{type(error).__name__}[/]\n\n{error}",
                title="request failed", title_align="left",
                box=box.ROUNDED, border_style="red", padding=(1, 2)))
        console.print()


if __name__ == "__main__":
    main()
