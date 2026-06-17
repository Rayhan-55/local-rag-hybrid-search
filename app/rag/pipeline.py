
from __future__ import annotations

from datetime import date
from typing import Optional

import httpx

from app.config import get_settings
from app.embeddings.embedder import embed_query
from app.vectorstore import chroma_store

SYSTEM_PROMPT = (
    "You are a careful document question-answering assistant for a bilingual "
    "(Bangla/English) archive. Answer ONLY using the provided context "
    "passages. If the answer is not in the context, say you could not find it "
    "in the documents. Reply in the SAME language as the user's question. "
    "Cite the source filename(s) you used."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        blocks.append(
            f"[Passage {i} | file: {m.get('filename')} | page: {m.get('page')} "
            f"| lang: {m.get('language')}]\n{c['text']}"
        )
    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return (
        f"{SYSTEM_PROMPT}\n\n=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n=== ANSWER ==="
    )


def _ollama_generate(prompt: str) -> str:
    s = get_settings()
    url = f"{s.ollama_base_url}/api/generate"
    payload = {
        "model": s.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    with httpx.Client(timeout=s.llm_timeout_seconds) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json().get("response", "").strip()


def retrieve(
    *,
    question: str,
    top_k: Optional[int] = None,
    language: Optional[str] = None,
    doc_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    s = get_settings()
    q_emb = embed_query(question)
    return chroma_store.query(
        query_embedding=q_emb,
        top_k=top_k or s.top_k,
        language=language,
        doc_type=doc_type,
        date_from=date_from,
        date_to=date_to,
    )


def answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return ("No matching passages were found under the selected filters, "
                "so I cannot answer from the documents.")
    prompt = _build_prompt(question, chunks)
    return _ollama_generate(prompt)
