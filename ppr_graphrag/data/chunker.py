"""Simple token-window document chunking."""

from __future__ import annotations

from ppr_graphrag.core.hashing import make_id
from ppr_graphrag.data.schema import Document


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        chunk = " ".join(tokens[start : start + chunk_size])
        if not chunk:
            continue
        chunks.append(chunk)
    return chunks


def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Document]:
    chunk_texts = chunk_text(doc.text, chunk_size, chunk_overlap)
    if len(chunk_texts) == 1 and chunk_texts[0] == doc.text:
        return [doc]
    documents: list[Document] = []
    step = chunk_size - chunk_overlap
    for i, text in enumerate(chunk_texts):
        start = i * step
        documents.append(
            Document(
                doc_id=make_id("chunk", {"doc_id": doc.doc_id, "start": start, "text": text}),
                title=doc.title,
                text=text,
                metadata={**doc.metadata, "source_doc_id": doc.doc_id, "chunk_start": start},
            )
        )
    return documents


def chunk_corpus(corpus: list[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> list[Document]:
    chunks: list[Document] = []
    for doc in corpus:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks
