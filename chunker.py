"""Text chunking utilities."""
import hashlib
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _chunk_id(chunk: Document) -> str:
    """
    Content-addressed chunk ID.

    Stable across re-indexes and independent of where the project lives on disk,
    so a file can be re-chunked and its old rows deleted by ID. A positional
    ``chunk_{i}`` would shift whenever any earlier document changed.
    """
    source = os.path.basename(str(chunk.metadata.get("source", "")))
    page = str(chunk.metadata.get("page", chunk.metadata.get("slide", "")))
    payload = f"{source}|{page}|{chunk.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def split_documents(
    documents: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks and assign stable, unique IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    unique: List[Document] = []
    seen: set = set()
    for chunk in chunks:
        if not chunk.page_content.strip():
            continue
        cid = _chunk_id(chunk)
        if cid in seen:
            # Identical text from the same page (repeated boilerplate) would
            # collide on ID in the vector store anyway; keep one copy.
            continue
        seen.add(cid)
        chunk.metadata["chunk_id"] = cid
        unique.append(chunk)

    return unique
