# Local OCR & Dynamic RAG System (Bangla / English)

A **fully local** document-processing and Retrieval-Augmented Generation pipeline.
Upload scanned PDFs or images containing **Bangla, English, or both**, extract text
with a local OCR engine, embed and store it in a local vector database, then ask
natural-language questions answered by a **local LLM (Ollama)** — with **hybrid
search** that combines semantic similarity with strict manual metadata filters
(date, type, language).

> **Privacy guarantee:** no document content ever leaves the host. OCR, embeddings,
> the vector store, and the LLM all run on your machine. The only network calls are
> to `localhost:11434` (your own Ollama).

---

## 1. Architecture

```
                       ┌──────────────────────────────────────────────┐
   Upload (PDF/IMG) ──►│  Ingestion                                   │
                       │   PyMuPDF render ─► OCR (Tesseract/Surya)     │
                       │   born-digital pages use embedded text layer  │
                       └───────────────┬──────────────────────────────┘
                                       │ full text + per-page conf
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │  Language detect (script ratio) + metadata    │
                       │  Bilingual chunker (danda/overlap aware)       │
                       │  Embed: BAAI/bge-m3 (local, multilingual)      │
                       └───────────────┬──────────────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │  ChromaDB (local, persistent)                 │
                       │  vectors + metadata {lang, type, date_epoch}  │
                       └───────────────┬──────────────────────────────┘
                                       ▲ query_embedding + WHERE filter
   Query + filters ────────────────────┘
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │  Hybrid retrieve  (filter ∩ vector top-k)     │
                       │  Grounded prompt ─► Ollama (qwen2.5)           │
                       │  Answer in the user's language + citations     │
                       └──────────────────────────────────────────────┘
```

**Stack**

| Layer        | Choice                          | Why                                              |
|--------------|---------------------------------|--------------------------------------------------|
| API          | FastAPI + Uvicorn               | async, typed, auto OpenAPI docs at `/docs`       |
| OCR          | Tesseract (`ben+eng`), Surya opt| local, Bangla pack available; pluggable interface |
| PDF render   | PyMuPDF (fitz)                  | no system poppler dependency                     |
| Embeddings   | `BAAI/bge-m3`                   | multilingual dense vectors, shared Bn/En space   |
| Vector DB    | ChromaDB (persistent)           | local, metadata `where` filters = hybrid search  |
| LLM          | Ollama (`qwen2.5:7b`)           | strong multilingual/Bangla, runs locally         |
| UI           | single static HTML page         | upload + filter + query demo, no build step      |

---

## 2. Quick start

### Option A — Docker (recommended)

```bash
cp .env.example .env
docker compose up --build          # builds app + starts Ollama
# pull the model into the ollama container (first run only):
docker exec -it rag-ollama ollama pull qwen2.5:7b
```
Open <http://localhost:8000>.

### Option B — Local (no Docker)

```bash
bash scripts/setup.sh              # installs tesseract+ben, venv, deps, pulls model
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
Open <http://localhost:8000>. API docs at <http://localhost:8000/docs>.

**Prereqs for local run:** Python 3.11+, [Ollama](https://ollama.com),
and Tesseract with the Bangla data file (`ben.traineddata`). The setup script
installs these on Debian/Ubuntu and macOS (brew); on Windows install Tesseract
from the UB-Mannheim build (tick the **Bengali** language pack) and Ollama from
ollama.com.

**Model size note:** the default is `qwen2.5:7b` (~4.7 GB, needs ~5–6 GB free
RAM). On a lighter machine, pull `qwen2.5:3b` instead and set
`OLLAMA_MODEL=qwen2.5:3b` in your `.env`. The model name in `.env` must exactly
match the one you pulled with `ollama pull`, or Ollama returns a 404.

---

## 3. API

| Method | Path                  | Purpose                                        |
|--------|-----------------------|------------------------------------------------|
| POST   | `/documents/upload`   | multipart: `file`, `doc_type`, `doc_date`, `language_override` |
| GET    | `/documents`          | list indexed documents                         |
| DELETE | `/documents/{id}`     | remove a document and its chunks               |
| POST   | `/search`             | `{query, filters:{language,doc_type,date_from,date_to}, generate_answer}` |
| GET    | `/health`             | engine/model status                            |

Example hybrid query:

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{
  "query": "এই নোটিশের শেষ তারিখ কবে?",
  "filters": {"language": "ben", "doc_type": "notice",
              "date_from": "2026-01-01", "date_to": "2026-12-31"},
  "generate_answer": true
}'
```

---

## 4. Must Explain

### 4.1 Local OCR choice, trade-offs, and Bangla baseline

**Default: Tesseract 5 (`--oem 1`, LSTM) with `lang=ben+eng`.** It is fully local,
ships an official Bangla model (`ben.traineddata`), installs in seconds, and runs
on CPU — the right default for a "must run locally" assessment.

Bangla is hard for OCR because of **conjuncts (যুক্তাক্ষর)**, vowel signs (মাত্রা/কার),
and the connecting headline stroke (মাত্রা). To protect recall we:
- render PDF pages at **300 DPI** (thin matra strokes vanish at 150 DPI);
- convert to grayscale, apply a **gentle** denoise (aggressive denoise eats matra),
  and **Otsu binarisation** to normalise uneven scan lighting;
- run a single `ben+eng` pass so bilingual pages are read together.

**Baseline expectation (printed/clean scans):** Tesseract typically lands around
**75–90% character accuracy** on clean printed Bangla, dropping on low-DPI,
skewed, or photocopied pages and on dense conjuncts. We surface a **mean
per-word confidence** from `image_to_data` per page as a runtime accuracy proxy
(reported in the upload response and visible in logs).

**Trade-off / escape hatch:** Because Bangla accuracy is the weak spot, OCR is
behind a pluggable `OCREngine` interface. Set `OCR_ENGINE=surya` to switch to
**Surya**, a transformer OCR that generally beats Tesseract on noisy/complex
Bangla layouts — at the cost of heavier compute (ideally a GPU). One env var,
zero code changes elsewhere. A further optimisation already implemented: if a
PDF page has an **embedded text layer** (born-digital), we use it directly and
skip OCR entirely — faster and lossless.

### 4.2 Chunking strategy and embedding model for a bilingual corpus

**Embeddings — `BAAI/bge-m3`.** A bilingual archive must let an English query
find Bangla passages and vice-versa, so both languages need to live in **one
shared vector space**. bge-m3 is explicitly multilingual (100+ languages,
strong on Bangla and English), produces L2-normalised 1024-dim dense vectors
(cosine ranking), and runs offline via `sentence-transformers`. Alternatives
considered: `intfloat/multilingual-e5-large` (also good, similar size);
English-only models like `bge-small-en` were rejected because they collapse
Bangla into noise.

**Chunking — character windows with sentence-aware boundaries and overlap.**
- *Character-based, not token-based.* Most subword tokenizers shatter Bangla
  into many sub-tokens, making token-count windows wildly uneven across the two
  languages. Character windows (`CHUNK_SIZE=900`) give predictable, fair sizes
  in both scripts.
- *Sentence-aware splitting.* We split on paragraph breaks and on terminators
  including the **Bangla danda `।`** plus `?!.`, so chunks rarely cut a sentence
  in half. Only a single oversized sentence is hard-sliced.
- *Sliding overlap (`CHUNK_OVERLAP=150`).* A fact straddling a boundary stays
  retrievable from at least one chunk, which measurably helps QA recall.

### 4.3 How manual metadata filtering works with vector similarity (hybrid)

The strict filters and the semantic search are **not** two separate passes that
we intersect in Python — they execute **together inside ChromaDB**:

1. **Hard filter (a guarantee).** The user's filters are translated into a Chroma
   `where` predicate (`app/vectorstore/chroma_store.py::_build_where`):
   - `language` → `{"language": {"$eq": ...}}`
   - `doc_type` → `{"doc_type": {"$eq": ...}}`
   - date range → `{"doc_date_epoch": {"$gte": from}}` and `{"$lte": to}`
   Dates are stored twice: ISO string for display, and **integer epoch-days**
   so Chroma can do fast numeric range filtering. A document outside the filter
   **can never** appear in results.
2. **Vector ranking (relevance).** Within the surviving set, Chroma ranks by
   cosine similarity between the query embedding (bge-m3) and chunk embeddings,
   returning `top_k`.

So: **filters decide *eligibility*, similarity decides *ordering*.** The
`/search` response returns both the generated answer and the ranked passages
with their scores and metadata, so the filtering is auditable in the UI.

---

## 5. Repo layout

```
local-rag-hybrid-search/
│
├── app/                      # ── BACKEND (FastAPI) ──
│   ├── main.py               # app entry; mounts routes, serves static UI
│   ├── config.py             # all tunables (env-overridable)
│   ├── schemas.py            # pydantic request/response models
│   ├── ocr/                  # OCR layer (pluggable)
│   │   ├── base.py           #   OCREngine interface
│   │   ├── tesseract_engine.py
│   │   ├── surya_engine.py   #   optional
│   │   └── __init__.py       #   engine factory
│   ├── ingestion/
│   │   ├── pdf_utils.py      # PDF→image render + language detect
│   │   └── chunker.py        # bilingual chunker (danda + overlap)
│   ├── embeddings/
│   │   └── embedder.py       # bge-m3 (multilingual)
│   ├── vectorstore/
│   │   └── chroma_store.py   # ChromaDB + hybrid metadata filter
│   ├── rag/
│   │   └── pipeline.py       # retrieve + Ollama generate
│   └── routes/
│       ├── documents.py      # upload / list / delete
│       └── search.py         # hybrid search endpoint
│
├── frontend/                 # ── FRONTEND (static UI) ──
│   └── index.html            # upload + filters + query, single page
│
├── scripts/
│   └── setup.sh              # local (non-Docker) setup
├── sample_data/              # test documents
│
├── Dockerfile                # backend image + tesseract-ocr-ben
├── docker-compose.yml        # app + ollama
├── requirements.txt          # Python deps
├── .env.example              # env template
├── .gitignore
└── README.md
```

The backend is everything under `app/` (the full FastAPI pipeline); the
frontend is the static UI under `frontend/`. They live in one repository
because the API also serves the single-page UI.

---


```
