
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.config import get_settings
from app.ocr import get_ocr_engine
from app.ocr.base import PageText

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


_BANGLA = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass
class ExtractedDoc:
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def mean_confidence(self) -> float:
        vals = [p.mean_confidence for p in self.pages if p.mean_confidence >= 0]
        return sum(vals) / len(vals) if vals else -1.0


def detect_language(text: str) -> str:
    """Cheap script-ratio language tag: 'ben', 'eng', or 'mixed'."""
    bn = len(_BANGLA.findall(text))
    en = len(_LATIN.findall(text))
    total = bn + en
    if total == 0:
        return "eng"
    bn_ratio = bn / total
    if bn_ratio > 0.85:
        return "ben"
    if bn_ratio < 0.15:
        return "eng"
    return "mixed"


def _ocr_image_file(path: str, page: int = 1) -> PageText:
    return get_ocr_engine().image_to_text(path, page=page)


def extract_document(file_path: str) -> ExtractedDoc:
    s = get_settings()
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTS:
        return ExtractedDoc(pages=[_ocr_image_file(file_path, page=1)])

    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext}")

    pages: list[PageText] = []
    doc = fitz.open(file_path)
    zoom = s.pdf_render_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(doc, start=1):
            embedded = page.get_text("text").strip()
            if len(embedded) >= 40:  # born-digital page: trust text layer
                pages.append(PageText(page=i, text=embedded, mean_confidence=-1.0))
                continue
            # Scanned page -> rasterise + OCR
            pix = page.get_pixmap(matrix=matrix)
            img_path = str(Path(tmp) / f"page_{i}.png")
            pix.save(img_path)
            pages.append(_ocr_image_file(img_path, page=i))
    doc.close()
    return ExtractedDoc(pages=pages)
