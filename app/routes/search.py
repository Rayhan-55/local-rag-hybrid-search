"""Hybrid RAG search route: semantic query + strict manual metadata filters."""
from __future__ import annotations

from fastapi import APIRouter

from app.rag import pipeline
from app.schemas import RetrievedChunk, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest):
    f = req.filters
    chunks = pipeline.retrieve(
        question=req.query,
        top_k=req.top_k,
        language=f.language,
        doc_type=f.doc_type,
        date_from=f.date_from,
        date_to=f.date_to,
    )

    retrieved = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            document_id=c["metadata"]["document_id"],
            filename=c["metadata"].get("filename", ""),
            text=c["text"],
            score=c["score"],
            language=c["metadata"].get("language", ""),
            doc_type=c["metadata"].get("doc_type", ""),
            doc_date=c["metadata"].get("doc_date"),
            page=c["metadata"].get("page"),
        )
        for c in chunks
    ]

    generated = pipeline.answer(req.query, chunks) if req.generate_answer else None

    return SearchResponse(
        query=req.query,
        answer=generated,
        chunks=retrieved,
        used_filters=f,
    )
