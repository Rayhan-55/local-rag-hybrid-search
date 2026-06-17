#for api input output
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    language: str
    doc_type: str
    doc_date: Optional[str]
    pages: int
    chunks_indexed: int
    char_count: int
    ocr_engine: str


class DocumentMeta(BaseModel):
    document_id: str
    filename: str
    language: str
    doc_type: str
    doc_date: Optional[str]
    pages: int
    chunks: int


class SearchFilters(BaseModel):
    """Strict manual filters applied *alongside* semantic similarity."""
    language: Optional[Literal["ben", "eng", "mixed"]] = None
    doc_type: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: SearchFilters = SearchFilters()
    top_k: Optional[int] = None
    generate_answer: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    language: str
    doc_type: str
    doc_date: Optional[str]
    page: Optional[int]


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str]
    chunks: list[RetrievedChunk]
    used_filters: SearchFilters
