# 🏗️ System Architecture — Solar Panel Chatbot

This document describes the technical architecture of the Solar Panel Smart Chatbot system.

---

## Overview

The system is a **Retrieval-Augmented Generation (RAG)** chatbot that combines:
1. A **vector knowledge base** (ChromaDB) for semantic search over documents
2. A **relational product database** (SQLite) for structured product queries
3. A **Large Language Model** (LLM) for natural language understanding and generation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                     │
│                                                                          │
│   ┌─────────────────────┐                                               │
│   │   index.html         │  Single-page application                      │
│   │   (Farsi RTL UI)     │  JavaScript + CSS                            │
│   └──────────┬───────────┘                                               │
│              │ HTTP/JSON (fetch API)                                      │
└──────────────┼───────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API LAYER (FastAPI)                              │
│                                                                          │
│   main.py                                                                │
│   ├── POST /ask          → Chat endpoint (orchestrates full pipeline)    │
│   ├── GET  /suggestions  → Starter questions                             │
│   ├── POST /feedback     → User feedback collection                      │
│   ├── GET  /products/*   → Product CRUD endpoints                        │
│   └── GET  /health       → System health check                           │
│                                                                          │
│   Middleware: CORS, Input Validation (Pydantic)                          │
└──────────────┬───────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CHATBOT ENGINE                                   │
│                                                                          │
│   chatbot.py (RAGChatbot class)                                          │
│                                                                          │
│   Pipeline Steps:                                                        │
│   1. classify_intent()    → Determine user's goal                        │
│   2. rewrite_query()      → Make query standalone from history           │
│   3. expand_query()       → Correct spelling, formalize                  │
│   4. retrieve_relevant_chunks() → Semantic search in ChromaDB            │
│   5. retrieve_product_info()    → Full-text search in SQLite             │
│   6. generate_response()        → LLM generates final answer             │
│                                                                          │
└────┬────────────────────────┬──────────────────────┬─────────────────────┘
     │                        │                      │
     ▼                        ▼                      ▼
┌────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ ChromaDB   │    │ SQLite            │    │ LLM API            │
│            │    │ (product_db.py)   │    │ (OpenAI-compatible)│
│ Vector     │    │                   │    │                    │
│ Store      │    │ products          │    │ Intent classify    │
│            │    │ categories        │    │ Query rewrite      │
│ Knowledge  │    │ feedback          │    │ Query expand       │
│ base docs  │    │ products_fts      │    │ Response generate  │
│            │    │                   │    │                    │
└────────────┘    └───────────────────┘    └────────────────────┘
```

---

## Component Details

### 1. Frontend (`frontend/index.html`)

| Aspect | Details |
|--------|---------|
| Type | Single-page application (SPA) |
| Framework | Vanilla JavaScript (no framework) |
| Styling | Custom CSS with dark theme |
| Direction | RTL (Right-to-Left) for Farsi |
| Features | Chat bubbles, product cards, typing indicator, feedback buttons |
| Communication | `fetch()` API calls to backend |

### 2. API Layer (`backend/main.py`)

| Aspect | Details |
|--------|---------|
| Framework | FastAPI (async Python) |
| Server | Uvicorn (ASGI) |
| Validation | Pydantic models for all request/response types |
| Middleware | CORS |
| Documentation | Auto-generated OpenAPI (Swagger) |

### 3. Chatbot Engine (`backend/chatbot.py`)

The core intelligence. Implements the RAG pipeline:

#### Intent Classification
- Uses LLM to classify user queries into:
  - `product_search` — User wants to find/browse products
  - `product_compare` — User wants to compare products
  - `general_support` — General questions about services, policies
  - `greeting` — Hello/hi messages
  - `farewell` — Goodbye messages

#### Query Rewriting
- Resolves pronouns and references using conversation history
- Example: "آن چطور است؟" → "پنل خورشیدی ۵۵۰ وات جی‌ای سولار چطور است؟"

#### Query Expansion
- Fixes Farsi spelling errors
- Formalizes colloquial language
- Returns original + corrected query for broader retrieval

#### Dual Retrieval
1. **ChromaDB** (semantic): Encodes query → finds similar document chunks
2. **SQLite FTS5** (keyword): Full-text search over product catalog

#### Response Generation
- Combines all context (documents + products + history)
- Sends to LLM with system prompt defining behavior rules
- Returns structured Farsi response

### 4. Product Database (`backend/product_db.py`)

| Table | Purpose |
|-------|---------|
| `categories` | Product categories (Solar Panels, Inverters, etc.) |
| `products` | Full product catalog with specs as JSON |
| `products_fts` | FTS5 virtual table for fast text search |
| `feedback` | User satisfaction ratings |

**Search Strategy**: SQLite FTS5 with triggers to keep the full-text index synchronized with the products table.

### 5. Configuration (`backend/config.py`)

All settings loaded from environment variables with sensible defaults. Uses `python-dotenv` to load `.env` file.

### 6. Knowledge Base (`ingest.ipynb` → `chroma_db/`)

- **Input**: Documents (PDF, TXT, MD) about solar panels
- **Process**: Chunk → Embed → Store
- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- **Storage**: ChromaDB persistent storage

---

## Data Flow: Complete Request Lifecycle

```
User types: "بهترین پنل خورشیدی برای خانه کدام است؟"
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 1. POST /ask                            │
│    body: {question, history, session_id}│
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 2. Intent Classification (LLM call #1)  │
│    Result: {                            │
│      intent: "product_search",          │
│      product_keywords: ["پنل خورشیدی"], │
│      category_hint: "پنل خورشیدی"       │
│    }                                    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 3. Query Rewrite (LLM call #2)          │
│    Input: query + last 6 history turns  │
│    Output: standalone query             │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 4. Query Expansion (LLM call #3)        │
│    Input: standalone query              │
│    Output: [original, corrected]        │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 5. ChromaDB Retrieval                   │
│    Encode queries → vector search       │
│    Return: top-6 relevant doc chunks    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 6. Product DB Search                    │
│    FTS5 search with keywords            │
│    Return: up to 5 matching products    │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 7. Response Generation (LLM call #4)    │
│    Context: docs + products + history   │
│    System prompt: Farsi support agent   │
│    Output: formatted Farsi response     │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│ 8. Return Response                      │
│    {answer, products[], intent,         │
│     session_id}                         │
└─────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | HTML/CSS/JS | ES6+ |
| API Framework | FastAPI | 0.100+ |
| ASGI Server | Uvicorn | 0.23+ |
| LLM Client | OpenAI Python SDK | 1.0+ |
| Vector DB | ChromaDB | 0.4+ |
| Embeddings | sentence-transformers | 2.2+ |
| Product DB | SQLite (via Python stdlib) | 3.35+ |
| Validation | Pydantic | 2.0+ |
| Config | python-dotenv | 1.0+ |

---

## Performance Characteristics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| First request | 5-15s | Loading embedding model into memory |
| Subsequent requests | 2-5s | LLM API latency dominates |
| Product search | <50ms | SQLite FTS5 is very fast |
| Vector search | 100-300ms | Depends on collection size |
| Memory usage | 1.5-2 GB | Mainly embedding model |
| Concurrent users | 10-50 | Limited by LLM API rate limits |

---

## Security Architecture

```
┌─────────────────────────────────────────┐
│ Security Measures                       │
│                                         │
│ ✓ API keys in env vars only             │
│ ✓ Pydantic input validation             │
│ ✓ CORS origin whitelist                 │
│ ✓ Parameterized SQL queries             │
│ ✓ Server-generated session IDs          │
│ ✓ Max query length (1000 chars)         │
│ ✓ No secrets in logs                    │
│ ✓ .gitignore for .env and .db files     │
└─────────────────────────────────────────┘
```

---

## Scaling Considerations

### Horizontal Scaling
- Backend is stateless (no in-memory sessions)
- SQLite limits write concurrency → migrate to PostgreSQL for high traffic
- ChromaDB can be replaced with Qdrant/Pinecone for distributed deployment

### Vertical Scaling
- Add more Gunicorn workers (`-w 8`)
- Use GPU for faster embedding generation
- Cache frequent queries/embeddings

### Cost Optimization
- Use `gpt-4o-mini` instead of `gpt-4o` (10x cheaper)
- Cache LLM responses for common questions
- Batch embedding generation during ingestion
