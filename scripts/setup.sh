#!/usr/bin/env bash
set -euo pipefail

# Local setup (no Docker). Tested on Ubuntu/Debian + macOS (brew).
echo ">> Installing Tesseract + Bangla language pack…"
if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng
elif command -v brew >/dev/null; then
  brew install tesseract tesseract-lang
else
  echo "!! Install Tesseract + the 'ben' language data manually."
fi

echo ">> Creating virtualenv…"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ">> Copying .env…"
[ -f .env ] || cp .env.example .env

echo ">> Pulling the local LLM via Ollama (make sure Ollama is installed: https://ollama.com)…"
if command -v ollama >/dev/null; then
  ollama pull qwen2.5:7b
else
  echo "!! Ollama not found. Install it, then run: ollama pull qwen2.5:7b"
fi

echo ">> Done. Start the API with:"
echo "   source .venv/bin/activate && uvicorn app.main:app --reload --port 8000"
echo "   then open http://localhost:8000"
