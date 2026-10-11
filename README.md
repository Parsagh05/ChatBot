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

On the first start the backend downloads the website (about 40 seconds) and builds the index (about 1 minute) in the background; chat answers "preparing" until it is ready. Later starts reuse the saved index.

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
| Website content | Admin dashboard → **Sync website** |
| A document | Put the file in `knowledge_base/raw_docs/`, then **Rebuild index** |
| Hand-written knowledge | Add entries to `knowledge_base/knowledge_base.json` (`chunk_id`, `chunk_text`, `metadata.source_title`, `metadata.source_url`), then **Rebuild index** |
| FAQ / startup suggestions | Edit `knowledge_base/QA.json`, then **Rebuild index** |

## Configuration

All settings are in `backend/.env` (see `backend/.env.example`). Settings changed in the admin dashboard are stored in `backend/data/runtime_settings.json` and override `.env`.

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
- The frontend keeps a short in-memory conversation history and sends it with each request; the backend does not store chats.
- Whether answers appear word by word depends on the LLM provider: some gateways buffer streamed responses and deliver them at once.
