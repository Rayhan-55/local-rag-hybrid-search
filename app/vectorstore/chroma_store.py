"""
Vector store: persistent ChromaDB.

This is where the *hybrid* search requirement is implemented. Chroma lets us
pass a `where` clause (structured metadata predicate) into the same call that
does the vector similarity search. So retrieval works in two stages that run
together inside Chroma:

    1. Hard filter  -> keep only chunks whose metadata satisfies the user's
                       strict manual filters (language, doc_type, date range).
    2. Vector rank  -> among the survivors, rank by cosine similarity to the
                       query embedding and return the top_k.

This means a filter is a guarantee (a document outside the date range can
NEVER appear), while semantic relevance decides ordering within the allowed
set. Dates are stored both as ISO strings (for display) and as integer
epoch-days (for fast >=/<= range filtering, which Chroma supports on numbers).
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


def _epoch_days(d: date) -> int:
    return (d - date(1970, 1, 1)).days


@lru_cache
def _client() -> chromadb.ClientAPI:
    s = get_settings()
    return chromadb.PersistentClient(
        path=str(s.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _collection():
    s = get_settings()
    return _client().get_or_create_collection(
        name=s.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    *,
    document_id: str,
    filename: str,
    language: str,
    doc_type: str,
    doc_date: Optional[date],
    chunk_texts: list[str],
    embeddings: list[list[float]],
    pages: list[int],
) -> int:
    col = _collection()
    ids, metadatas = [], []
    for i, _ in enumerate(chunk_texts):
        ids.append(f"{document_id}:{i}")
        meta: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "language": language,
            "doc_type": doc_type,
            "page": pages[i] if i < len(pages) else 0,
            "chunk_index": i,
        }
        if doc_date is not None:
            meta["doc_date"] = doc_date.isoformat()
            meta["doc_date_epoch"] = _epoch_days(doc_date)
        metadatas.append(meta)
    col.add(ids=ids, documents=chunk_texts, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def _build_where(
    *,
    language: Optional[str],
    doc_type: Optional[str],
    date_from: Optional[date],
    date_to: Optional[date],
) -> Optional[dict]:
    """Translate user filters into a Chroma `where` predicate."""
    clauses: list[dict] = []
    if language:
        clauses.append({"language": {"$eq": language}})
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})
    if date_from:
        clauses.append({"doc_date_epoch": {"$gte": _epoch_days(date_from)}})
    if date_to:
        clauses.append({"doc_date_epoch": {"$lte": _epoch_days(date_to)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query(
    *,
    query_embedding: list[float],
    top_k: int,
    language: Optional[str] = None,
    doc_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    col = _collection()
    where = _build_where(
        language=language, doc_type=doc_type,
        date_from=date_from, date_to=date_to,
    )
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0],
        res["metadatas"][0], res["distances"][0],
    ):
        out.append({
            "chunk_id": cid,
            "text": doc,
            "metadata": meta,
            # cosine distance -> similarity score
            "score": round(1.0 - float(dist), 4),
        })
    return out


def list_documents() -> list[dict]:
    col = _collection()
    got = col.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in got["metadatas"] or []:
        did = meta["document_id"]
        if did not in docs:
            docs[did] = {
                "document_id": did,
                "filename": meta.get("filename"),
                "language": meta.get("language"),
                "doc_type": meta.get("doc_type"),
                "doc_date": meta.get("doc_date"),
                "pages": 0,
                "chunks": 0,
            }
        docs[did]["chunks"] += 1
        docs[did]["pages"] = max(docs[did]["pages"], int(meta.get("page", 0)))
    return list(docs.values())


def delete_document(document_id: str) -> None:
    _collection().delete(where={"document_id": {"$eq": document_id}})
