"""Load JSON knowledge base files and maintain the Chroma vector index."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from config import settings

logger = logging.getLogger(__name__)


def load_knowledge_documents() -> List[Dict[str, Any]]:
    """Load all JSON-backed RAG documents."""
    base_dir = Path(settings.KNOWLEDGE_BASE_DIR)
    documents: List[Dict[str, Any]] = []

    kb_path = base_dir / "knowledge_base.json"
    if kb_path.exists():
        data = json.loads(kb_path.read_text(encoding="utf-8"))
        for item in data:
            metadata = item.get("metadata") or {}
            chunk_id = str(item.get("chunk_id") or "")
            text = str(item.get("chunk_text") or "").strip()
            if not chunk_id or not text:
                continue
            documents.append(
                {
                    "id": f"kb::{chunk_id}",
                    "text": text,
                    "metadata": _clean_metadata(
                        {
                            **metadata,
                            "source_file": "knowledge_base.json",
                            "chunk_id": chunk_id,
                            "source_type": "knowledge_chunk",
                        }
                    ),
                }
            )
    else:
        logger.warning("Missing knowledge base file: %s", kb_path)

    qa_path = base_dir / "QA.json"
    if qa_path.exists():
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        for item in data:
            question_id = str(item.get("question_id") or "")
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not question_id or not question or not answer:
                continue
            documents.append(
                {
                    "id": f"qa::{question_id}",
                    "text": f"سوال: {question}\nپاسخ: {answer}",
                    "metadata": _clean_metadata(
                        {
                            "source_title": question,
                            "source_url": str(item.get("source_url") or ""),
                            "source_file": "QA.json",
                            "chunk_id": question_id,
                            "source_type": "faq",
                        }
                    ),
                }
            )
    else:
        logger.warning("Missing FAQ knowledge file: %s", qa_path)

    return documents


def load_suggested_questions(limit: int = 5) -> List[str]:
    """Load startup suggestions from QA.json."""
    qa_path = Path(settings.KNOWLEDGE_BASE_DIR) / "QA.json"
    if not qa_path.exists():
        return []
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    questions = [str(item.get("question") or "").strip() for item in data]
    return [question for question in questions if question][:limit]


def ensure_knowledge_base_index(embedding_model: Any, force: bool = False):
    """Create or refresh the Chroma collection from JSON files."""
    documents = load_knowledge_documents()
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection_name = settings.CHROMA_COLLECTION_NAME
    source_hash = _documents_hash(documents)

    collection = None
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = None

    if collection is not None and not force:
        metadata = collection.metadata or {}
        if metadata.get("source_hash") == source_hash and collection.count() == len(documents):
            logger.info("Knowledge base Chroma collection is up to date with %s documents.", len(documents))
            return collection

    if collection is not None:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "source": "knowledge_base_json",
            "source_hash": source_hash,
            "source_count": str(len(documents)),
        },
    )

    if not documents:
        logger.warning("Created empty Chroma collection; no JSON documents were loaded.")
        return collection

    batch_size = 32
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        texts = [doc["text"] for doc in batch]
        embeddings = embedding_model.encode(texts).tolist()
        collection.add(
            ids=[doc["id"] for doc in batch],
            documents=texts,
            metadatas=[doc["metadata"] for doc in batch],
            embeddings=embeddings,
        )

    logger.info("Indexed %s JSON knowledge documents into Chroma.", len(documents))
    return collection


def _documents_hash(documents: List[Dict[str, Any]]) -> str:
    payload = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in metadata.items()}
