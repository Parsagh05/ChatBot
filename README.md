# Solar Panel Smart Chatbot

Farsi-first RAG chatbot for solar product support. The backend uses FastAPI, ChromaDB, SQLite, and an OpenAI-compatible or Gemini LLM client. The frontend is a single RTL HTML chat UI.

## Current Inventory

The app builds its product database from `knowledge_base/knowledge_base.json` at startup. In the current inventory there are 11 product chunks across these categories:

- Solar panels
- Off-grid inverters
- Hybrid inverters
- Solar lights

Prices are not currently stored in the product JSON, so product cards show a contact-us price unless prices are added to the inventory/importer.

## Run Locally

Install dependencies:

```powershell
cd C:/Users/USER/Desktop/chatbot-main
python -m pip install -r backend/requirements.txt
```

Create or edit `backend/.env` from `backend/.env.example`, then start the backend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/frontend
python -m http.server 8080
```

Open `http://localhost:8080`.

## Configuration

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI-compatible API key. Also used as Gemini fallback if `GEMINI_API_KEY` is empty. |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint, for example TapSage or OpenAI. |
| `GEMINI_API_KEY` | Gemini API key, used when `LLM_MODEL` starts with `gemini`. |
| `GEMINI_API_ENDPOINT` | Gemini endpoint override. Keep this matched to your provider. |
| `LLM_MODEL` | Model name, for example `gpt-4o-mini` or a `gemini...` model. |
| `CHROMA_DB_PATH` | Persistent ChromaDB path, relative to `backend/` by default. |
| `KNOWLEDGE_BASE_DIR` | Folder containing `knowledge_base.json` and `QA.json`. |
| `PRODUCT_DB_PATH` | SQLite product/feedback database path. |
| `CORS_ORIGINS` | Comma-separated frontend origins. |

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/ask` | Main chat endpoint. |
| `GET` | `/suggestions` | Suggested startup questions from `QA.json`. |
| `POST` | `/feedback` | Save thumbs up/down feedback. |
| `GET` | `/products/search?q=...` | Search products. |
| `GET` | `/products/categories` | List product categories. |
| `GET` | `/products/category/{id}` | Products in one category. |
| `GET` | `/products/{id}` | Product details. |
| `GET` | `/health` | Backend health check. |

## Notes

- The frontend keeps only shallow in-memory conversation history and sends it with each `/ask` request.
- The backend does not persist full chat history by session ID.
- If the product database looks stale after changing JSON import logic, delete `backend/products.db` or call the importer with `force=True`.
- The frontend request timeout is 60 seconds; slow or hung LLM calls show a timeout message instead of leaving typing forever.
