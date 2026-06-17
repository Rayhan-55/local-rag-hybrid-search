
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PageText:
    page: int          # 1-indexed
    text: str
    mean_confidence: float  # 0-100, -1 if unknown


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def image_to_text(self, image_path: str, page: int = 1) -> PageText:
        """Run OCR on a single rasterised page image."""
        raise NotImplementedError
