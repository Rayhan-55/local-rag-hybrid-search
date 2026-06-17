
from __future__ import annotations

import re

from app.config import get_settings

# Sentence terminators incl. Bangla danda (।)
_SENT_SPLIT = re.compile(r"(?<=[।?!\.])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        parts.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return parts


def chunk_text(text: str) -> list[str]:
    s = get_settings()
    size, overlap = s.chunk_size, s.chunk_overlap
    sentences = _split_sentences(text)

    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        # Sentence longer than a whole window -> hard slice it.
        if len(sent) > size:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for j in range(0, len(sent), size - overlap):
                chunks.append(sent[j:j + size].strip())
            continue
        if len(buf) + len(sent) + 1 <= size:
            buf = f"{buf} {sent}".strip()
        else:
            chunks.append(buf.strip())
            # carry overlap tail into next buffer
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail} {sent}".strip()
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]
