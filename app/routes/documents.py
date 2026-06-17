"""Routes for uploading/ingesting documents and listing/deleting them."""
from __future__ import annotations

import shutil
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.embeddings.embedder import embed_passages
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_utils import detect_language, extract_document
from app.ocr import get_ocr_engine
from app.schemas import DocumentMeta, IngestResponse
from app.vectorstore import chroma_store

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@router.post("/upload", response_model=IngestResponse)
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    doc_date: Optional[str] = Form(None),
    language_override: Optional[str] = Form(None),
):
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED)}")

    document_id = uuid.uuid4().hex[:12]
    saved_path = Path(s.upload_dir) / f"{document_id}{ext}"
    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 1) OCR / text extraction (fully local)
    extracted = extract_document(str(saved_path))
    full_text = extracted.full_text
    if not full_text.strip():
        raise HTTPException(422, "No text could be extracted from the document.")

    # 2) Metadata
    language = language_override or detect_language(full_text)
    parsed_date: Optional[date] = None
    if doc_date:
        try:
            parsed_date = date.fromisoformat(doc_date)
        except ValueError:
            raise HTTPException(400, "doc_date must be ISO format YYYY-MM-DD")

    # 3) Chunk + embed (track which page each chunk roughly came from)
    chunks: list[str] = []
    chunk_pages: list[int] = []
    for pt in extracted.pages:
        for ch in chunk_text(pt.text):
            chunks.append(ch)
            chunk_pages.append(pt.page)
    if not chunks:
        raise HTTPException(422, "Document produced no chunks.")

    embeddings = embed_passages(chunks)

    # 4) Store in vector DB with metadata
    indexed = chroma_store.add_chunks(
        document_id=document_id,
        filename=file.filename or saved_path.name,
        language=language,
        doc_type=doc_type,
        doc_date=parsed_date,
        chunk_texts=chunks,
        embeddings=embeddings,
        pages=chunk_pages,
    )

    return IngestResponse(
        document_id=document_id,
        filename=file.filename or saved_path.name,
        language=language,
        doc_type=doc_type,
        doc_date=parsed_date.isoformat() if parsed_date else None,
        pages=len(extracted.pages),
        chunks_indexed=indexed,
        char_count=len(full_text),
        ocr_engine=get_ocr_engine().name,
    )


@router.get("", response_model=list[DocumentMeta])
async def list_docs():
    return [DocumentMeta(**d) for d in chroma_store.list_documents()]


@router.delete("/{document_id}")
async def delete_doc(document_id: str):
    chroma_store.delete_document(document_id)
    return {"deleted": document_id}
