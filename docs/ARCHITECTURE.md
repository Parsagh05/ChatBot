# 🏗️ System Architecture — Solar Panel Chatbot

This document describes the technical architecture of the Solar Panel Smart Chatbot system.

---

## Overview

The system is a **Retrieval-Augmented Generation (RAG)** chatbot that combines:
1. A **vector knowledge base** (ChromaDB) built automatically from the company website, documents and FAQ
2. A **relational product database** (SQLite) filled from the website's shop
3. A **Large Language Model** (OpenAI-compatible or Gemini) for understanding and generation
4. An **admin dashboard** for runtime configuration and knowledge-base maintenance

```
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ index.html (Farsi RTL chat)  │   │ admin.html (dashboard)       │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │ SSE / JSON                        │ JSON + Bearer token
               ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ main.py (FastAPI)            admin.py (/admin/* router)          │
│  /ask/stream  /ask            settings · status · sync · reindex │
│  /products/*  /feedback       background JobRunner               │
│ security.py: rate limiting, admin sessions                       │
└──────────────┬──────────────────────────────────┬───────────────┘
               ▼                                  ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ chatbot.py (RAGChatbot)      │   │ knowledge_store.py           │
│ 1. analyze_query (1 LLM call)│   │ KnowledgeIndex: collect →    │
│ 2. vector + product search   │◀──│ chunk → embed → Chroma,      │
│ 3. stream answer (LLM call)  │   │ swapped in when ready        │
│ 4. links.fix_links           │   └──────┬───────────┬───────────┘
└──────┬───────────────┬───────┘          │           │
       ▼               ▼                  ▼           ▼
┌────────────┐ ┌──────────────┐ ┌─────────────────┐ ┌───────────────────┐
│ LLM API    │ │ SQLite       │ │ site_crawler.py │ │ document_loader.py│
│ llm_client │ │ product_db   │ │ WordPress + Woo │ │ raw_docs: docx,   │
│            │ │ products_fts │ │ REST APIs       │ │ pdf, txt, md      │
└────────────┘ └──────────────┘ └─────────────────┘ └───────────────────┘
```

---

## Knowledge Base

### Sources (`knowledge_store.collect_sources`)

| Source | Loader | Notes |
| --- | --- | --- |
| Website pages and posts (articles, services, projects) | `site_crawler.py` → `/wp-json/wp/v2/pages`, `/posts` | HTML converted to text; empty pages (cart, forms) skipped |
| Website products | `/wp-json/wc/store/v1/products` + `/wp-json/wp/v2/product` | Store API for price, stock, images, category; WP API for the description text |
| Company documents | `document_loader.py` | `.docx` (paragraphs + tables), `.pdf` (PyMuPDF, reading order kept for RTL), `.txt`, `.md` |
| FAQ | `QA.json` | One chunk per question; also used for startup suggestions |
| Curated knowledge (optional) | `knowledge_base.json` | Hand-written chunks; exact duplicates of other sources are dropped |

The website is saved to `backend/data/site_snapshot.json`, so the server starts without network access. The first start with no snapshot runs a sync automatically.

### Chunking and indexing

- Text is split by paragraph, then sentence (`. ! ? ؟ ؛`), then word, and packed into chunks of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` characters of overlap. FAQ and curated entries stay whole.
- The source title and categories are prepended to each chunk **for embedding only**, so short chunks keep their context.
- The collection name contains a fingerprint of the embedding model and all chunks. An unchanged knowledge base reuses the stored index in seconds; any change builds a new collection.
- Builds run in a background job. The previous index keeps serving queries until the new one is complete, then it is swapped in and stale collections are deleted.

### Exact links (`links.py`)

Persian URLs are served percent-encoded by the site (`%d8%ae%d8%af...`). Hand-typed or model-written URLs often lose the scheme, a hyphen or the zero-width non-joiner. `LinkResolver` compares URLs on a canonical form (no scheme/`www`/trailing slash, decoded, NFC, ZWNJ/tatweel/separators removed, Arabic ي/ك → Persian ی/ک) and maps every spelling to the site's exact permalink. The model sees readable URLs in its context; its answer is post-processed so each URL becomes `[readable form](exact permalink)`.

---

## Request Lifecycle (`POST /ask/stream`)

```
1. Validate input (question ≤ 1000 chars, history ≤ 20 turns, roles user/assistant) and rate limit
2. analyze_query — one LLM call returning JSON:
   intent, standalone_query (resolved against history), corrected_query, product_keywords, category_hint
3. greeting / farewell → canned reply, no retrieval
4. Vector search with the original question + rewrite + correction; best TOP_K chunks overall
5. Product search (SQLite FTS5 with LIKE fallback)
6. SSE "meta" event: intent, products, sources (exact URLs)
7. Generation streamed as SSE "delta" events
8. SSE "done" event: full answer with links fixed to exact permalinks
```

Endpoints are plain `def` functions, so FastAPI runs them in its threadpool and a slow LLM call does not block other users.

---

## Configuration (`config.py`)

`Settings` loads defaults from `.env`, then applies overrides from `backend/data/runtime_settings.json`. The admin dashboard edits the keys in `EDITABLE_SETTINGS`:

- LLM settings → the LLM client is recreated immediately
- `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` → a background reindex starts
- Everything else (top-k, temperature, max tokens, rate limit) applies to the next request

API keys are never returned by the API; the dashboard only shows whether a key is set and its last four characters.

---

## Security

| Measure | Where |
| --- | --- |
| Admin password (`ADMIN_PASSWORD`), constant-time compare, 12-hour random session tokens | `security.py` |
| Login attempts limited to 5 per minute per client | `security.py` |
| Chat/feedback rate limit per client (`RATE_LIMIT_PER_MINUTE`) | `security.py` |
| Pydantic validation of all request bodies and settings ranges | `main.py`, `admin.py` |
| All model output and data escaped before rendering; only `http(s)` links rendered | `index.html`, `admin.html` |
| Parameterized SQL | `product_db.py` |
| CORS origin whitelist | `main.py` |
| `.env`, `backend/data/`, databases and `knowledge_base/` excluded from git | `.gitignore` |

The rate limiter and admin sessions are in memory, which suits a single server process. Running several workers would need a shared store (e.g. Redis).

---

## Feedback

The chat UI sends 👍 (rating 5) or 👎 (rating 1) with the exact question and answer of that message to `POST /feedback`; rows are stored in the `feedback` table of `products.db`. The dashboard lists them through `GET /admin/feedback` (newest first, filter `up`/`down`, paged), with totals.

---

## Deployment and Tests

- `Dockerfile`: Python 3.12 slim, CPU-only PyTorch, non-root user, all data under `/data`, one Uvicorn worker with `--proxy-headers`.
- `docker-compose.yml`: `backend` + `web` (nginx). nginx serves `frontend/`, replaces `config.js` so the pages call `/api`, and proxies `/api/` to the backend with buffering off for SSE. It overwrites `X-Forwarded-For` with the real client address so rate limits cannot be bypassed.
- `backend/tests/`: pytest suite (links, chunking, document loading, product import, settings, chat/stream API, rate limits, admin auth, settings validation, feedback) using temp databases, a fake LLM and a fake index.
- `.github/workflows/ci.yml`: runs the tests, validates compose and nginx config, builds the image and imports the app inside it.

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| Frontend | HTML/CSS/vanilla JS, RTL, Vazirmatn font |
| API | FastAPI + Uvicorn |
| LLM | OpenAI Python SDK; Gemini REST API via httpx |
| Vector DB | ChromaDB (persistent, cosine HNSW) |
| Embeddings | sentence-transformers (`paraphrase-multilingual-mpnet-base-v2` by default) |
| Product DB | SQLite with FTS5 |
| Ingestion | httpx, BeautifulSoup, python-docx, PyMuPDF |

---

## Performance (measured on a CPU-only Windows PC)

| Operation | Typical value |
| --- | --- |
| Website sync (≈170 pages/posts, 10 products) | ~40 s |
| Full index build (≈280 sources → ≈570 chunks) | ~50 s |
| Startup with an existing index | ~3 s |
| Vector search | 30–90 ms |
| Full answer (2 LLM calls, Gemini flash-lite via gateway) | ~4 s |
