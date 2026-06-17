
from __future__ import annotations

from .base import OCREngine, PageText


class SuryaEngine(OCREngine):
    name = "surya"

    def __init__(self, languages: str = "ben+eng"):
        # Lazy import so Tesseract-only installs don't need Surya's deps.
        from PIL import Image  # noqa
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Surya is not installed. `pip install surya-ocr` or set "
                "OCR_ENGINE=tesseract in your .env."
            ) from e
        self._Image = Image
        self.det = DetectionPredictor()
        self.rec = RecognitionPredictor()
        # Surya autodetects script, but we hint Bangla+English.
        self.langs = ["bn", "en"]

    def image_to_text(self, image_path: str, page: int = 1) -> PageText:
        image = self._Image.open(image_path).convert("RGB")
        predictions = self.rec([image], [self.langs], self.det)
        lines = []
        confs = []
        for pred in predictions:
            for line in pred.text_lines:
                lines.append(line.text)
                if line.confidence is not None:
                    confs.append(line.confidence * 100.0)
        mean_conf = sum(confs) / len(confs) if confs else -1.0
        return PageText(page=page, text="\n".join(lines).strip(),
                        mean_confidence=mean_conf)
