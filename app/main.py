from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import documents, search

app = FastAPI(
    title="Local OCR + Dynamic RAG (Bangla/English)",
    version="1.0.0",
    description="Fully local document QA with hybrid metadata + semantic search.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/health")
async def health():
    s = get_settings()
    return {
        "status": "ok",
        "ocr_engine": s.ocr_engine,
        "embedding_model": s.embedding_model,
        "llm": s.ollama_model,
    }


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")



if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")