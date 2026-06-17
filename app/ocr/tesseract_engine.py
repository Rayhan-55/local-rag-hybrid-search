
from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .base import OCREngine, PageText


def _preprocess(image_path: str) -> Image.Image:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to PIL if cv2 can't read (e.g. odd formats)
        return Image.open(image_path).convert("L")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise gently — aggressive denoise eats Bangla matra strokes.
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Otsu binarisation handles uneven scan lighting well.
    _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binar)


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, languages: str = "ben+eng"):
        self.languages = languages
        # psm 3 = fully automatic page segmentation (good default for docs)
        self.config = "--oem 1 --psm 3"

    def image_to_text(self, image_path: str, page: int = 1) -> PageText:
        pil_img = _preprocess(image_path)
        text = pytesseract.image_to_string(
            pil_img, lang=self.languages, config=self.config
        )
        # Confidence proxy
        data = pytesseract.image_to_data(
            pil_img, lang=self.languages, config=self.config,
            output_type=pytesseract.Output.DICT,
        )
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
        mean_conf = float(np.mean(confs)) if confs else -1.0
        return PageText(page=page, text=text.strip(), mean_confidence=mean_conf)
