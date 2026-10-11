"""Build the knowledge base from the website, raw documents and QA.json, and maintain the Chroma index."""

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb

from config import settings
from document_loader import load_raw_documents
from links import LinkResolver
from site_crawler import load_snapshot

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?؟!؛])\s+")


# ─── Sources ─────────────────────────────────────────────────────────────────


def load_faq() -> List[Dict[str, Any]]:
    qa_path = Path(settings.KNOWLEDGE_BASE_DIR) / "QA.json"
    if not qa_path.exists():
        logger.warning("Missing FAQ knowledge file: %s", qa_path)
        return []
    try:
        data = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read %s: %s", qa_path, exc)
        return []
    return [item for item in data if item.get("question") and item.get("answer")]


def load_curated() -> List[Dict[str, Any]]:
    """Optional hand-written chunks in knowledge_base.json: [{chunk_id, chunk_text, metadata: {source_title, source_url}}]."""
    path = Path(settings.KNOWLEDGE_BASE_DIR) / "knowledge_base.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read %s: %s", path, exc)
        return []
    curated = []
    for item in data:
        metadata = item.get("metadata") or {}
        text = str(item.get("chunk_text") or "").strip()
        if not item.get("chunk_id") or not text:
            continue
        curated.append(
            {
                "id": f"curated:{item['chunk_id']}",
                "source_type": "curated",
                "title": str(metadata.get("source_title") or item["chunk_id"]).strip(),
                "url": str(metadata.get("source_url") or ""),
                "categories": [str(metadata["parent_category"])] if metadata.get("parent_category") else [],
                "text": text,
            }
        )
    return curated


def load_suggested_questions(limit: int = 5) -> List[str]:
    """Load startup suggestions from QA.json."""
    return [str(item["question"]).strip() for item in load_faq()][:limit]


def product_text(product: Dict[str, Any]) -> str:
    parts = [product["name"]]
    if product.get("categories"):
        parts.append("دسته‌بندی: " + "، ".join(product["categories"]))
    for key in ("short_description", "description"):
        if product.get(key):
            parts.append(product[key])
    for name, value in (product.get("attributes") or {}).items():
        parts.append(f"{name}: {value}")
    return "\n".join(parts)


def collect_sources() -> List[Dict[str, Any]]:
    """Every source document: website pages/posts/products, raw documents, curated JSON and FAQ entries."""
    sources: List[Dict[str, Any]] = []
    snapshot = load_snapshot()
    if snapshot:
        sources.extend(snapshot.get("documents", []))
        for product in snapshot.get("products", []):
            sources.append(
                {
                    "id": f"product:{product['id']}",
                    "source_type": "product",
                    "title": product["name"],
                    "url": product.get("url", ""),
                    "categories": product.get("categories", []),
                    "text": product_text(product),
                }
            )
    else:
        logger.warning("No website snapshot yet; run a website sync from the admin dashboard.")

    sources.extend(load_raw_documents())

    for item in load_faq():
        sources.append(
            {
                "id": f"faq:{item.get('question_id') or item['question']}",
                "source_type": "faq",
                "title": str(item["question"]).strip(),
                "url": str(item.get("source_url") or ""),
                "categories": [],
                "text": f"سوال: {str(item['question']).strip()}\nپاسخ: {str(item['answer']).strip()}",
            }
        )
    sources.extend(load_curated())

    # drop exact duplicates (e.g. curated JSON entries copied from QA.json); the first occurrence wins
    unique, seen = [], set()
    for source in sources:
        key = re.sub(r"\s+", " ", source["text"]).strip()
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def build_link_resolver(sources: List[Dict[str, Any]]) -> LinkResolver:
    """Resolver over all source URLs. Website permalinks come first in sources, so they win over hand-typed FAQ URLs."""
    return LinkResolver((source.get("url", "") for source in sources), site_url=settings.SITE_URL)


# ─── Chunking ────────────────────────────────────────────────────────────────


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split on paragraphs, then sentences, then words; pack into overlapping chunks of ~chunk_size chars."""
    units: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > chunk_size:
                cut = sentence.rfind(" ", 0, chunk_size)
                cut = cut if cut > chunk_size // 2 else chunk_size
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)

    chunks: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_size:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            if tail and len(tail) < len(current) and " " in tail:
                tail = tail[tail.index(" ") + 1 :]  # start the overlap on a word boundary
            current = f"{tail}\n{unit}" if tail else unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def build_chunks(sources: List[Dict[str, Any]], resolver: LinkResolver) -> List[Dict[str, Any]]:
    chunks = []
    for source in sources:
        # FAQ answers and curated chunks are already sized by hand; keep each one whole
        pieces = (
            [source["text"]]
            if source["source_type"] in ("faq", "curated")
            else split_text(source["text"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        )
        url = resolver.resolve(source.get("url", ""))
        categories = "، ".join(source.get("categories") or [])
        for i, piece in enumerate(pieces):
            header = source["title"] + (f" ({categories})" if categories else "")
            chunks.append(
                {
                    "id": f"{source['id']}#{i}",
                    "text": piece,
                    # title/category are prepended for embedding only, to give short chunks context
                    "embed_text": f"{header}\n{piece}",
                    "metadata": {
                        "source_id": source["id"],
                        "source_type": source["source_type"],
                        "source_title": source["title"],
                        "source_url": url,
                        "categories": categories,
                    },
                }
            )
    return chunks


# ─── Index ───────────────────────────────────────────────────────────────────


class KnowledgeIndex:
    """Embedding model + Chroma collection, rebuilt in the background and swapped in atomically."""

    def __init__(self) -> None:
        self.embedding_model = None
        self.embedding_model_name = ""
        self.collection = None
        self.resolver = LinkResolver(site_url=settings.SITE_URL)
        self.stats: Dict[str, Any] = {}
        self._build_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.collection is not None and self.embedding_model is not None

    def build(self, force: bool = False, progress=lambda message: None) -> Dict[str, Any]:
        """Load or rebuild the index for the current settings and sources."""
        with self._build_lock:
            progress("در حال خواندن منابع...")
            sources = collect_sources()
            resolver = build_link_resolver(sources)
            chunks = build_chunks(sources, resolver)

            model_name = settings.EMBEDDING_MODEL
            model = self.embedding_model if model_name == self.embedding_model_name else None
            if model is None:
                progress(f"در حال بارگذاری مدل embedding ({model_name})...")
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)

            fingerprint = _fingerprint(model_name, chunks)
            name = f"{settings.CHROMA_COLLECTION_NAME}_{fingerprint[:12]}"
            client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

            collection = None
            if not force:
                try:
                    existing = client.get_collection(name)
                    if existing.count() == len(chunks):
                        collection = existing
                        logger.info("Chroma collection %s is up to date (%s chunks).", name, len(chunks))
                except Exception:
                    collection = None

            if collection is None:
                try:
                    client.delete_collection(name)
                except Exception:
                    pass
                collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine", "embedding_model": model_name})
                batch_size = 64
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    progress(f"در حال ساخت ایندکس: {start + len(batch)} از {len(chunks)} بخش")
                    collection.add(
                        ids=[c["id"] for c in batch],
                        documents=[c["text"] for c in batch],
                        metadatas=[c["metadata"] for c in batch],
                        embeddings=model.encode([c["embed_text"] for c in batch]).tolist(),
                    )
                logger.info("Indexed %s chunks into %s.", len(chunks), name)

            # swap in the new index, then drop stale collections
            self.embedding_model, self.embedding_model_name = model, model_name
            self.collection, self.resolver = collection, resolver
            self.stats = _stats(sources, chunks)
            for other in client.list_collections():
                other_name = getattr(other, "name", other)
                if other_name != name and str(other_name).startswith(settings.CHROMA_COLLECTION_NAME):
                    client.delete_collection(other_name)
            return self.stats

    def search(self, queries: List[str], top_k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Top-k chunks across all queries, best first, without duplicates."""
        model, collection = self.embedding_model, self.collection
        if model is None or collection is None or not queries:
            return []
        best: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
        embeddings = model.encode(queries).tolist()
        result = collection.query(query_embeddings=embeddings, n_results=top_k)
        for ids, docs, metas, distances in zip(result["ids"], result["documents"], result["metadatas"], result["distances"]):
            for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
                if chunk_id not in best or distance < best[chunk_id][0]:
                    best[chunk_id] = (distance, doc, meta)
        ranked = sorted(best.values(), key=lambda item: item[0])[:top_k]
        return [(doc, meta) for _, doc, meta in ranked]


def _fingerprint(model_name: str, chunks: List[Dict[str, Any]]) -> str:
    payload = json.dumps(
        [model_name, [(c["id"], c["embed_text"], c["metadata"]) for c in chunks]],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stats(sources: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    for source in sources:
        by_type[source["source_type"]] = by_type.get(source["source_type"], 0) + 1
    snapshot = load_snapshot() or {}
    return {
        "sources": len(sources),
        "chunks": len(chunks),
        "sources_by_type": by_type,
        "site_fetched_at": snapshot.get("fetched_at"),
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }
