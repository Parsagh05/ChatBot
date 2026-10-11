# Solar Panel Chatbot Setup

## Requirements

- Python 3.10+
- Internet access for dependency installation, the first website sync, the one-time embedding model download (~1.1 GB) and LLM API calls
- At least 4 GB RAM, 8 GB recommended because the embedding model runs locally

## Install

```powershell
cd C:/Users/USER/Desktop/chatbot-main
python -m venv .venv
.venv\Scripts\python -m pip install -r backend/requirements.txt
```

## Configure

Copy the example environment file and edit it:

```powershell
copy backend\.env.example backend\.env
```

Minimum configuration:

```env
ADMIN_PASSWORD=choose-a-strong-password
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.tapsage.com/openai/v1
LLM_MODEL=gpt-4o-mini
```

For Gemini-style model names, provide a Gemini key and ensure the endpoint matches your provider:

```env
GEMINI_API_KEY=your-gemini-key
GEMINI_API_ENDPOINT=https://api.tapsage.com
LLM_MODEL=gemini-3.1-flash-lite
```

Keys and models can also be changed later from the admin dashboard without restarting.

## Run

Backend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/backend
..\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/frontend
python -m http.server 8080
```

- Chat: `http://localhost:8080`
- Admin dashboard: `http://localhost:8080/admin.html` (password = `ADMIN_PASSWORD`)

## Run with Docker

```bash
cp backend/.env.example backend/.env
docker compose up -d --build
docker compose logs -f backend   # first start: model download and website sync
```

Open `http://localhost:8080` and `http://localhost:8080/admin.html`. To update the code: `git pull && docker compose up -d --build`. Data survives rebuilds in the `chatbot-data` volume.

## Run the Tests

```powershell
cd C:/Users/USER/Desktop/chatbot-main/backend
..\.venv\Scripts\python -m pip install -r requirements-dev.txt
..\.venv\Scripts\python -m pytest
```

## Data Files

| Path | Kind | Purpose |
| --- | --- | --- |
| `knowledge_base/raw_docs/` | you maintain | Company documents (`.docx`, `.pdf`, `.txt`, `.md`) |
| `knowledge_base/QA.json` | you maintain | FAQ answers and startup suggestions |
| `knowledge_base/knowledge_base.json` | you maintain, optional | Hand-written knowledge chunks |
| `backend/data/site_snapshot.json` | generated | Last website sync |
| `backend/data/runtime_settings.json` | generated | Settings saved from the admin dashboard (contains API keys; never commit) |
| `backend/products.db` | generated | Products (from the website) and feedback |
| `chroma_db/` | generated | Vector index |

## Troubleshooting

- **Chat says it is preparing the knowledge base:** the first sync and index build are still running; see the admin dashboard status.
- **Embedding model download stalls:** delete the `*.incomplete` files under `%USERPROFILE%\.cache\huggingface\hub\models--sentence-transformers--...\blobs` and start again; downloads resume.
- **Admin dashboard says it is disabled:** set `ADMIN_PASSWORD` in `backend/.env` and restart the backend.
- **"Too many requests":** raise `RATE_LIMIT_PER_MINUTE` in the dashboard.
- **Persian text looks corrupted:** hard refresh the browser and make sure source files are saved as UTF-8.
- Backend API docs are available at `http://localhost:8000/docs` when the server is running.
