# Solar Panel Smart Chatbot

Farsi-first RAG chatbot for TabEnergy (تابان انرژی مازند سو) customer support. The backend uses FastAPI, ChromaDB, SQLite, and an OpenAI-compatible or Gemini LLM. The frontend is an RTL chat UI plus an admin dashboard.

## Features

- **Knowledge base built automatically** from every page, article, service, project and product on [tabenergy.ir](https://tabenergy.ir) (through its WordPress/WooCommerce APIs), the documents in `knowledge_base/raw_docs/` (`.docx`, `.pdf`, `.txt`, `.md`), the FAQ in `knowledge_base/QA.json`, and optional hand-written chunks in `knowledge_base/knowledge_base.json`.
- **Exact website links.** Every link the bot shows opens the site's exact permalink and is displayed in readable Persian (e.g. `tabenergy.ir/خدمات/774/`). Links the model writes are matched back to the real URL even when the scheme, a hyphen or an invisible zero-width character is missing.
- **Admin dashboard** (`frontend/admin.html`) to change API keys, LLM provider/model, embedding model, chunk size/overlap, top-k, temperature, max tokens and rate limit at runtime, test the LLM connection, sync the website and rebuild the index. Index rebuilds run in the background; the old index keeps answering until the new one is ready.
- **Streaming answers** over Server-Sent Events, with one query-analysis call (intent + standalone rewrite + spelling fix) before the answer.
- **Product cards** with image, stock and a link to the product page; source links under each answer.
- **Feedback view** in the dashboard: every 👍/👎 rating with the question and answer, filterable to see what users disliked.
- **Security:** password-protected admin API, escaped rendering in the UI, per-client rate limiting, validated input, non-blocking endpoints.

## Project Structure

```
backend/
  main.py             FastAPI app: chat, products, feedback, health
  admin.py            /admin API: settings, status, feedback, sync/reindex jobs
  chatbot.py          RAG pipeline: query analysis, retrieval, streamed answer
  knowledge_store.py  Collects sources, chunks them, maintains the Chroma index
  site_crawler.py     Downloads tabenergy.ir through its WordPress/WooCommerce APIs
  document_loader.py  Reads raw_docs (.docx, .pdf, .txt, .md)
  links.py            Exact-URL resolver and readable Persian display
  llm_client.py       OpenAI-compatible and Gemini clients (with streaming)
  product_db.py       SQLite products (with full-text search) and feedback
  security.py         Rate limiting and admin sessions
  config.py           .env settings plus dashboard overrides
  tests/              pytest suite
frontend/
  index.html          Chat UI
  admin.html          Admin dashboard
  config.js           Backend address used by both pages
deploy/               nginx config for Docker
knowledge_base/       Your data — not in this repository (see below)
ingest.ipynb          Legacy Phase 3 ingestion notebook, no longer used
```

## Knowledge Base Folder

`knowledge_base/` is kept out of git because it holds company documents. Create it next to `backend/` before running:

```
knowledge_base/
  raw_docs/            company documents (.docx, .pdf, .txt, .md)
  QA.json              FAQ: [{"question_id", "question", "answer", "source_url"}]
  knowledge_base.json  optional hand-written chunks
```

Without it the bot still works from the website alone, but has no documents, FAQ answers or startup suggestions.

## Run Locally

Install dependencies (a virtual environment is recommended):

```powershell
cd C:/Users/USER/Desktop/chatbot-main
python -m venv .venv
.venv\Scripts\python -m pip install -r backend/requirements.txt
```

Create `backend/.env` from `backend/.env.example`, set `ADMIN_PASSWORD` and an LLM key, then start the backend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/backend
..\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/frontend
python -m http.server 8080
```

- Chat: `http://localhost:8080`
- Admin dashboard: `http://localhost:8080/admin.html`

On the first start the backend downloads the embedding model (~1.1 GB, once), then the website (about 40 seconds), and builds the index (about 1 minute), all in the background; chat answers "preparing" until it is ready. Later starts reuse the saved index and take a few seconds.

## Run with Docker

```bash
cp backend/.env.example backend/.env   # set ADMIN_PASSWORD and an LLM key
docker compose up -d --build
```

- Chat: `http://localhost:8080`, admin dashboard: `http://localhost:8080/admin.html`
- nginx serves the frontend and proxies `/api/` to the backend (streaming passes through unbuffered).
- The index, databases, site snapshot, dashboard settings and the embedding model cache live in the `chatbot-data` volume; `knowledge_base/` is mounted read-only from the host.
- The first start downloads the embedding model (~1.1 GB) and the website; follow it with `docker compose logs -f backend`.

## Tests

```powershell
cd backend
..\.venv\Scripts\python -m pip install -r requirements-dev.txt
..\.venv\Scripts\python -m pytest
```

The tests use temporary databases, a fake LLM and a fake index, so they need no network, API key or model download. GitHub Actions (`.github/workflows/ci.yml`) runs them on every push, and also validates the compose and nginx files and builds the Docker image.

## Updating the Knowledge Base

| To add or update | Do this |
| --- | --- |
| Website content | Publish it on tabenergy.ir, then admin dashboard → **Sync website** |
| A document | Put the file in `knowledge_base/raw_docs/`, then **Rebuild index** |
| Hand-written knowledge | Add entries to `knowledge_base/knowledge_base.json`, then **Rebuild index** |
| FAQ / startup suggestions | Edit `knowledge_base/QA.json`, then **Rebuild index** |

The bot never picks up changes on its own: after any change, press **Sync website** (for the site) or **Rebuild index** (for files). A rebuild takes about a minute, and the bot keeps answering from the previous index meanwhile.

### Adding New Documents

Pick the way that fits the document:

| The document is… | Best way |
| --- | --- |
| Public information that belongs on the website (new product, service, project, article) | Publish it on the site and press **Sync website**; answers then link to the page |
| A clean Word/PDF/text file (catalog, company profile, report) | Drop it into `raw_docs/` (option A) |
| Messy, scanned, very long, or a mix of topics | Have Claude turn it into curated JSON entries (option B) |
| A short question with a fixed answer | Add it to `QA.json` (option C) |

**Option A: drop the file into `raw_docs/`**

1. Save it as `.docx`, `.pdf`, `.txt` or `.md`. Old `.doc` files are ignored; open them in Word and save as `.docx`.
2. Name the file after its content, because the name becomes the source title the bot cites. Leading numbers and a `Rev..` suffix are removed: `06- کاتالوگ اینورترها- Rev01.docx` → «کاتالوگ اینورترها».
3. Copy it into `knowledge_base/raw_docs/` (with Docker, the same folder on the host).
4. Admin dashboard → **Rebuild index**.
5. Check: the knowledge-base card shows one more «سند», and a question about the document is answered from it.

Things to know:
- **Scanned PDFs (images of pages) contain no text** and are skipped, with a "No text extracted" line in the backend log. Run OCR on them first, or use option B.
- To **update** a document, replace the file and rebuild. To **remove** one, delete the file and rebuild.
- Documents have no web address, so answers from them show no link. If the same content is on the website, prefer the website.
- Text inside images, charts and text boxes is not read; tables are.

**Option B: curated entries in `knowledge_base.json`, written with Claude**

Each entry is one self-contained piece of knowledge:

```json
{
  "chunk_id": "inverter_catalog_growatt_spf6000",
  "chunk_text": "سانورتر GROWATT مدل SPF 6000 ES Plus با توان خروجی ۶۰۰۰ وات و ولتاژ ۴۸ ولت، شارژ کنترلر MPPT صد آمپری دارد و تا ۶ دستگاه قابل موازی شدن است.",
  "metadata": {
    "source_title": "کاتالوگ اینورترها",
    "source_url": "https://tabenergy.ir/product-category/off-grid-inverte/",
    "parent_category": "اینورتر منفصل از شبکه"
  }
}
```

Prompt you can give Claude together with the document and the current `knowledge_base.json`:

> Convert the attached document into new entries for my chatbot's `knowledge_base.json` and add them to the existing array. Rules: write in formal Persian; one topic per entry; each `chunk_text` must make sense on its own (repeat the product or service name instead of "it"), about 300–800 characters; keep every number, model name and specification exactly as in the document; `chunk_id` must be unique, lowercase English with underscores; `source_title` is the document's title; `source_url` is the matching tabenergy.ir page if there is one, otherwise empty; `parent_category` only if it fits. Don't change or remove existing entries. Return the whole file as valid JSON.

Then save the file (UTF-8), make sure it is still valid JSON, and press **Rebuild index**. Entries are never split, so keep each one short. An entry whose text exactly matches an existing one is skipped automatically.

**Option C: a fixed answer in `QA.json`**

Add `{"question_id": "faq_43", "question": "...", "answer": "...", "source_url": "..."}` with a new unique `question_id`, then **Rebuild index**. The first five questions in the file are the suggestions shown when the chat opens.

Links in `source_url` may be typed the readable way (`tabenergy.ir/خدمات/774/`); the bot maps them to the site's exact address.

## Configuration

All settings are in `backend/.env` (see `backend/.env.example`). Settings changed in the admin dashboard are stored in `backend/data/runtime_settings.json` (in Docker: inside the `chatbot-data` volume) and override `.env`. Everything except `ADMIN_PASSWORD`, `CORS_ORIGINS`, `CHROMA_COLLECTION_NAME` and the paths can be changed from the dashboard.

| Variable | Description |
| --- | --- |
| `ADMIN_PASSWORD` | Admin dashboard password; the dashboard is disabled while empty. |
| `LLM_PROVIDER` | `auto`, `openai` or `gemini`. `auto` uses Gemini for `gemini*` model names. |
| `LLM_MODEL` | Chat model, e.g. `gpt-4o-mini` or a `gemini...` model. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI-compatible API, e.g. TapSage or OpenAI. |
| `GEMINI_API_KEY` / `GEMINI_API_ENDPOINT` | Gemini API or a compatible gateway. |
| `EMBEDDING_MODEL` | sentence-transformers model used for the vector index. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk length and overlap in characters. |
| `TOP_K` | Number of chunks given to the model per question. |
| `TEMPERATURE` / `MAX_TOKENS` | Generation settings. |
| `SITE_URL` | Website to sync. |
| `RATE_LIMIT_PER_MINUTE` | Chat requests allowed per client per minute. |
| `KNOWLEDGE_BASE_DIR` | Folder with `raw_docs/`, `QA.json` and optional `knowledge_base.json`. |
| `CHROMA_DB_PATH` / `PRODUCT_DB_PATH` / `DATA_DIR` | Generated data locations. |
| `CHROMA_COLLECTION_NAME` | Prefix of the Chroma collection names. |
| `CORS_ORIGINS` | Comma-separated frontend origins. |

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/ask/stream` | Chat, streamed as Server-Sent Events (`meta`, `delta`…, `done`). |
| `POST` | `/ask` | Chat, full answer in one response. |
| `GET` | `/suggestions` | Suggested startup questions from `QA.json`. |
| `POST` | `/feedback` | Save thumbs up/down feedback. |
| `GET` | `/products/search?q=...` | Search products. |
| `GET` | `/products/categories` | List product categories. |
| `GET` | `/products/category/{id}` | Products in one category. |
| `GET` | `/products/{id}` | Product details. |
| `GET` | `/health` | Backend health check. |
| `POST` | `/admin/login`, `/admin/logout` | Admin session. |
| `GET` / `PUT` | `/admin/settings` | Read / change runtime settings. |
| `GET` | `/admin/status` | LLM, index, products and background job status. |
| `GET` | `/admin/feedback?rating=down` | User ratings, newest first (`rating`: `up`, `down`, or omitted for all). |
| `POST` | `/admin/sync`, `/admin/reindex`, `/admin/test-llm` | Admin actions. |

All `/admin/*` endpoints except login require `Authorization: Bearer <token>`.

## Notes

- Prices are not published on the website, so product cards show a contact-us price.
- The frontend keeps a short in-memory conversation history and sends it with each request. The backend does not store conversations; only answers a user rates with 👍/👎 are saved (question, answer, rating) for the feedback view.
- Whether answers appear word by word depends on the LLM provider: some gateways buffer streamed responses and deliver them at once.
- Rate limits, admin sessions and background jobs live in process memory, so run a single backend worker.

## Versions

Each development phase is tagged in git:

| Tag | Summary |
| --- | --- |
| `v1.0-phase1` | Command-line RAG chatbot in a notebook, keyword matching over `.docx` files |
| `v2.0-phase2` | JSON knowledge base, sentence-transformers embeddings, ChromaDB, LLM query correction |
| `v3.0-phase3` | FastAPI backend, web chat UI, conversation-aware query rewriting |
| `v4.0-phase4` | Solar product database, Gemini support, Farsi RTL UI with product cards |
| `v5.0` | Admin dashboard, automatic knowledge base from the website and documents, exact links, streaming, security, Docker, tests, CI |
