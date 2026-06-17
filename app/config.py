from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Storage ----
    data_dir: Path = BASE_DIR / "storage"
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"

   
    ocr_engine: str = "tesseract"
   
    ocr_languages: str = "ben+eng"
    
    pdf_render_dpi: int = 300

    
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu" 

    
    chunk_size: int = 900         
    chunk_overlap: int = 150       

   
    collection_name: str = "documents"

  
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"  
    llm_timeout_seconds: int = 120

   
    top_k: int = 5

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.chroma_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
