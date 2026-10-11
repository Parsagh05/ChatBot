# Solar Panel Chatbot Setup

## Requirements

- Python 3.10+
- Internet access for dependency installation and LLM API calls
- At least 4 GB RAM, 8 GB recommended because the embedding model is loaded locally

## Install

```powershell
cd C:/Users/USER/Desktop/chatbot-main
python -m pip install -r backend/requirements.txt
```

Using a virtual environment is recommended, but the app also runs with global Python packages if that is your local workflow.

## Configure

Copy the example environment file and edit the keys:

```powershell
copy backend/.env.example backend/.env
```

Common configuration:

```env
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

## Run

Backend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd C:/Users/USER/Desktop/chatbot-main/frontend
python -m http.server 8080
```

Open `http://localhost:8080`.

## Data Files

- `knowledge_base/knowledge_base.json`: RAG chunks and product chunks
- `knowledge_base/QA.json`: FAQ questions and answers used for suggestions
- `backend/products.db`: generated SQLite product database and feedback store
- `chroma_db/`: generated ChromaDB vector index

Current inventory is generated from 11 product chunks. Prices are not present in the JSON inventory, so the UI displays a contact-us price.

## Troubleshooting

- If the chat stays on typing, the frontend now times out after 60 seconds and shows a timeout message.
- If Persian text appears corrupted, hard refresh the browser and make sure source files are saved as UTF-8.
- If products look stale, remove `backend/products.db` and restart the backend so it reimports the JSON inventory.
- Backend docs are available at `http://localhost:8000/docs` when the server is running.
