# Chatbot Project

This repository contains the code for a chatbot, developed across multiple phases, evolving from a simple keyword-based system to a more sophisticated RAG application with a web interface.

## Description

A Python-based RAG (Retrieval-Augmented Generation) chatbot designed to answer questions based on a provided knowledge base. It leverages vector embeddings for semantic search, integrates with OpenAI-compatible language models, and features a FastAPI backend with a web frontend. 🗣️ 🌐

---

## Accessing Project Phases

This project uses Git tags to mark the end of each development phase. You can check out the specific code for any phase using `git checkout <tag_name>`.

* **Phase 1 (`phase-1-cli`)**: The initial baseline version. 🌱
    * **Knowledge Source:** Loads text directly from `.docx` files located in a `knowledge_docs` directory.
    * **Chunking:** Splits documents into chunks based on blank lines (double newlines) within the `.docx` files.
    * **Retrieval Method:** Implements a **naive keyword matching** strategy. It splits the query and chunks into words and finds the chunk with the highest number of overlapping words. Only the single best-matching chunk is used as context.
    * **Response Generation:** Uses the `openai` library, configured to potentially connect to custom endpoints (like `api.metisai.ir`), to generate answers based on the retrieved chunk and the user's query. The prompt includes basic instructions for the LLM to act as a customer service assistant.
    * **User Interface:** Interaction occurs directly within the `RAG_Chatbot.ipynb` Jupyter Notebook through an input loop.
    * **Dependencies:** `openai`, `python-docx`, `python-dotenv`.

* **Phase 2 (`phase-2`)**: Enhanced retrieval, structured data, and CLI. 🚀
    * **Knowledge Source:** Shifts to a structured `knowledge_base.json` file, where each entry contains a `chunk_id`, `chunk_text`, and associated `metadata` (like source title and URL).
    * **Ingestion Pipeline:** Introduces a dedicated ingestion script (`Vector_Database_Ingestion.ipynb`). It uses `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`) to generate vector embeddings for each chunk's text.
    * **Vector Database:** Stores the embeddings, original text, and metadata persistently in a **`ChromaDB`** vector store located in the `./chroma_db` directory.
    * **Retrieval Method:** Implements **semantic search**. Queries are embedded using the same `sentence-transformers` model, and `ChromaDB` retrieves the `n_results` (e.g., 10) most similar chunks based on vector distance.
    * **Query Enhancement:** Adds an **LLM-based query correction/expansion** step (`expand_query`). It uses `gpt-4o-mini` (via `api.tapsage.com`) to correct spelling/grammar mistakes in the user's query. Retrieval is performed using *both* the original and corrected queries to broaden the search.
    * **User Interface:** Provides a dedicated **Command-Line Interface (CLI)** application (`Chatbot_CLI_Application.ipynb`) for a more standard user interaction flow.
    * **Dependencies:** `chromadb`, `sentence-transformers`, `openai`, `python-dotenv`.

* **Phase 3 (`phase-3`)**: Backend API, Web UI, and Conversation History. ✨
    * **Architecture:** Implements a client-server model.
    * **Backend:** A **FastAPI** application (`backend/main.py`) serves the chatbot logic via an `/ask` REST API endpoint. It encapsulates the core chatbot functions within a `RAGChatbot` class (`backend/chatbot.py`). Uses Uvicorn for serving.
    * **Frontend:** A simple **Web UI** (`frontend/index.html`) using HTML and vanilla JavaScript allows users to interact with the chatbot through their browser. It communicates with the FastAPI backend via `fetch` requests. Includes basic chat styling, message display, and a typing indicator.
    * **Conversation Context:** Introduces **query rewriting** (`rewrite_query` in `backend/chatbot.py`). Before retrieval, the LLM reframes the user's latest question using the recent conversation history (last few turns) to make it a standalone, contextually complete query.
    * **Ingestion Improvement:** The ingestion script (`ingest.ipynb`) now **enriches the text *before* embedding**. It prepends metadata (like source title and parent category) to the chunk text to potentially create more contextually rich embeddings for better retrieval, while still storing the original text in the database for display/generation.
    * **Generation Enhancement:** The prompt sent to the LLM for the final answer now includes formatted context chunks *with their metadata* (source, URL) and the recent conversation history, allowing for more informed and potentially source-citing responses.
    * **Dependencies:** Managed via `requirements.txt`, including `fastapi`, `uvicorn`, `chromadb`, `sentence-transformers`, `openai`, `python-dotenv`. CORS middleware is configured in FastAPI.
