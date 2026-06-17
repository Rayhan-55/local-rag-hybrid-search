"""OCR factory — returns the engine selected by OCR_ENGINE env var."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.ocr.base import OCREngine, PageText  # noqa: F401 (re-export)


@lru_cache
def get_ocr_engine() -> OCREngine:
    s = get_settings()
    if s.ocr_engine == "surya":
        from app.ocr.surya_engine import SuryaEngine
        return SuryaEngine(languages=s.ocr_languages)
    from app.ocr.tesseract_engine import TesseractEngine
    return TesseractEngine(languages=s.ocr_languages)
