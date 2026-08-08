"""
Persistent, content-addressed document index.

Chroma holds the vectors; a JSON manifest next to it records the SHA-256 of every
source file that has been indexed and which chunk IDs came from it. On startup we
compare the manifest against what is on disk and only re-embed files that are new
or have actually changed.
"""
import hashlib
import json
import os
import shutil
from typing import Dict, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from chunker import split_documents
from embeddings import DISTANCE_SPACE, get_embeddings
from loaders import discover, load_file

MANIFEST_NAME = "manifest.json"
COLLECTION_NAME = "documents"

# Chroma rejects very large single writes; add in batches.
_ADD_BATCH = 512

# A directory is only safe to wipe if it is empty or already looks like one of ours.
_INDEX_MARKERS = ("chroma.sqlite3", MANIFEST_NAME)


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Fingerprint of everything that changes how chunks are produced or embedded.

    If this differs from the manifest, no existing row is reusable and we rebuild
    from scratch — otherwise you would silently mix vectors from two models.
    """
    payload = "|".join([
        embedding_model, DISTANCE_SPACE, "normalized",
        str(chunk_size), str(chunk_overlap),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, MANIFEST_NAME)


def _read_manifest(persist_dir: str) -> Dict:
    try:
        with open(_manifest_path(persist_dir), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_manifest(persist_dir: str, manifest: Dict) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    tmp = _manifest_path(persist_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp, _manifest_path(persist_dir))


def _reset(persist_dir: str) -> None:
    """
    Wipe the index directory.

    Guarded: CHROMA_PERSIST_DIR comes from the environment, and an unguarded
    rmtree on a value like "." would delete the project.
    """
    resolved = os.path.abspath(persist_dir)
    if os.path.isdir(resolved):
        entries = os.listdir(resolved)
        if entries and not any(marker in entries for marker in _INDEX_MARKERS):
            raise ValueError(
                f"Refusing to delete {resolved!r}: it is not empty and does not "
                f"look like an index directory (expected one of {_INDEX_MARKERS})."
            )
        shutil.rmtree(resolved)
    os.makedirs(resolved, exist_ok=True)


def _open_store(persist_dir: str, embedding_model: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(embedding_model),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": DISTANCE_SPACE},
    )


def _all_chunks(store: Chroma) -> List[Document]:
    """
    Rehydrate every stored chunk, for the BM25 corpus.

    BM25 has no on-disk form here, so it is rebuilt each start — but from the
    vector store rather than by re-parsing and re-embedding the source files.
    """
    raw = store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    docs.sort(key=lambda d: d.metadata.get("chunk_id", ""))
    return docs


def load_or_build_index(
    docs_path: str,
    persist_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[Chroma, List[Document]]:
    """
    Bring the on-disk index in sync with docs_path and return (store, chunks).

    Unchanged files are never re-read, re-chunked, or re-embedded.
    """
    signature = _signature(embedding_model, chunk_size, chunk_overlap)
    manifest = _read_manifest(persist_dir)

    if manifest.get("signature") != signature:
        if manifest:
            print("Index settings changed — rebuilding from scratch")
        _reset(persist_dir)
        manifest = {"signature": signature, "files": {}}

    store = _open_store(persist_dir, embedding_model)
    tracked: Dict[str, Dict] = manifest.setdefault("files", {})

    on_disk = {
        os.path.relpath(path, os.path.abspath(docs_path)): (path, _file_hash(path))
        for path in discover(docs_path)
    }

    removed = [key for key in tracked if key not in on_disk]
    changed = [
        key for key, (_, digest) in on_disk.items()
        if tracked.get(key, {}).get("hash") != digest
    ]

    # Drop stale rows first so a re-chunked file cannot leave orphans behind.
    for key in removed + changed:
        stale_ids = tracked.get(key, {}).get("chunk_ids", [])
        if stale_ids:
            store.delete(ids=stale_ids)
        tracked.pop(key, None)

    for key in changed:
        path, digest = on_disk[key]
        chunks = split_documents(load_file(path), chunk_size, chunk_overlap)
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        for start in range(0, len(chunks), _ADD_BATCH):
            batch = chunks[start:start + _ADD_BATCH]
            store.add_documents(batch, ids=ids[start:start + _ADD_BATCH])
        tracked[key] = {"hash": digest, "chunk_ids": ids}
        print(f"  indexed {key} ({len(ids)} chunks)")

    for key in removed:
        print(f"  removed {key}")

    _write_manifest(persist_dir, manifest)

    if not changed and not removed:
        print("Index up to date — no re-embedding needed")

    return store, _all_chunks(store)
