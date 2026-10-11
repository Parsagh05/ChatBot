"""Product database management using SQLite and the JSON knowledge base."""

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


PRODUCT_PREFIX = "product_"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    db_path = Path(settings.PRODUCT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the product and feedback database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            name_fa TEXT NOT NULL,
            description TEXT,
            description_fa TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_chunk_id TEXT UNIQUE,
            source_url TEXT,
            name TEXT NOT NULL,
            name_fa TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            description TEXT,
            description_fa TEXT,
            price REAL,
            currency TEXT DEFAULT 'تومان',
            in_stock INTEGER DEFAULT 1,
            specifications TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
        """
    )

    _ensure_column(cursor, "products", "source_chunk_id", "TEXT")
    _ensure_column(cursor, "products", "source_url", "TEXT")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_products_source_chunk_id ON products(source_chunk_id)")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER CHECK(rating >= 1 AND rating <= 5),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS import_state (
            source TEXT PRIMARY KEY,
            source_hash TEXT NOT NULL,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
            name, name_fa, description, description_fa, specifications,
            content='products', content_rowid='id'
        )
        """
    )

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
            INSERT INTO products_fts(rowid, name, name_fa, description, description_fa, specifications)
            VALUES (new.id, new.name, new.name_fa, new.description, new.description_fa, new.specifications);
        END
        """
    )

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
            INSERT INTO products_fts(products_fts, rowid, name, name_fa, description, description_fa, specifications)
            VALUES ('delete', old.id, old.name, old.name_fa, old.description, old.description_fa, old.specifications);
        END
        """
    )

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
            INSERT INTO products_fts(products_fts, rowid, name, name_fa, description, description_fa, specifications)
            VALUES ('delete', old.id, old.name, old.name_fa, old.description, old.description_fa, old.specifications);
            INSERT INTO products_fts(rowid, name, name_fa, description, description_fa, specifications)
            VALUES (new.id, new.name, new.name_fa, new.description, new.description_fa, new.specifications);
        END
        """
    )

    conn.commit()
    conn.close()


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, column_type: str) -> None:
    existing = {row["name"] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def seed_sample_data() -> None:
    """Backward-compatible alias for the old hardcoded seeder."""
    import_products_from_knowledge_base()


def import_products_from_knowledge_base(force: bool = False) -> int:
    """Import product rows from knowledge_base.json into SQLite."""
    knowledge_path = Path(settings.KNOWLEDGE_BASE_DIR) / "knowledge_base.json"
    if not knowledge_path.exists():
        logger.warning("Product import skipped; missing %s", knowledge_path)
        return 0

    raw_json = knowledge_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT source_hash FROM import_state WHERE source = ?", ("knowledge_base_products",))
    row = cursor.fetchone()
    if row and row["source_hash"] == source_hash and not force:
        cursor.execute("SELECT COUNT(*) FROM products")
        count = int(cursor.fetchone()[0])
        conn.close()
        return count

    data = json.loads(raw_json)
    products = [_product_from_chunk(item) for item in data if _is_product_chunk(item)]
    products = [product for product in products if product]

    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM categories")
    cursor.execute("DELETE FROM products_fts")

    category_ids: Dict[str, int] = {}
    for product in products:
        category_name = product["category_name_fa"]
        if category_name not in category_ids:
            cursor.execute(
                """
                INSERT INTO categories (name, name_fa, description, description_fa)
                VALUES (?, ?, ?, ?)
                """,
                (category_name, category_name, None, None),
            )
            category_ids[category_name] = int(cursor.lastrowid)

        cursor.execute(
            """
            INSERT INTO products (
                source_chunk_id, source_url, name, name_fa, category_id,
                description, description_fa, price, currency, in_stock, specifications, image_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product["source_chunk_id"],
                product["source_url"],
                product["name"],
                product["name_fa"],
                category_ids[category_name],
                product["description"],
                product["description_fa"],
                product["price"],
                product["currency"],
                product["in_stock"],
                json.dumps(product["specifications"], ensure_ascii=False),
                product["image_url"],
            ),
        )

    cursor.execute(
        """
        INSERT INTO import_state (source, source_hash, imported_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source) DO UPDATE SET
            source_hash = excluded.source_hash,
            imported_at = CURRENT_TIMESTAMP
        """,
        ("knowledge_base_products", source_hash),
    )

    conn.commit()
    conn.close()
    logger.info("Imported %s products from %s", len(products), knowledge_path)
    return len(products)


def _is_product_chunk(item: Dict[str, Any]) -> bool:
    chunk_id = str(item.get("chunk_id", ""))
    return chunk_id.startswith(PRODUCT_PREFIX)


def _product_from_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    title = str(metadata.get("source_title") or "").strip()
    text = str(item.get("chunk_text") or "").strip()
    category = str(metadata.get("parent_category") or "محصولات").strip()
    source_url = str(metadata.get("source_url") or "").strip()

    return {
        "source_chunk_id": str(item.get("chunk_id") or title),
        "source_url": source_url,
        "name": _latin_name(title) or title,
        "name_fa": title,
        "category_name_fa": category,
        "description": text,
        "description_fa": text,
        "price": None,
        "currency": "تومان",
        "in_stock": 1,
        "specifications": _extract_specifications(text, title, category, source_url),
        "image_url": None,
    }


def _latin_name(title: str) -> str:
    parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9.+-]*)*", title)
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_specifications(text: str, title: str, category: str, source_url: str) -> Dict[str, str]:
    specs: Dict[str, str] = {
        "عنوان": title,
        "دسته‌بندی": category,
    }
    if source_url:
        specs["لینک منبع"] = source_url

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        key, value = line.lstrip("-").split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            specs[key] = value

    return specs


def search_products(query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search products with FTS and a LIKE fallback for short or complex queries."""
    query = (query or "").strip()
    if not query:
        return []

    rows = _search_products_fts(query, category, limit)
    if not rows:
        rows = _search_products_like(query, category, limit)
    return [_row_to_product(row) for row in rows]


def _search_products_fts(query: str, category: Optional[str], limit: int) -> List[sqlite3.Row]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", query, flags=re.UNICODE)
    tokens = [token for token in tokens if len(token) > 1]
    if not tokens:
        return []
    match_query = " OR ".join(f'"{token}"' for token in tokens[:8])

    conn = get_connection()
    cursor = conn.cursor()
    try:
        params: List[Any] = [match_query]
        category_clause = ""
        if category:
            category_clause = "AND (c.name_fa LIKE ? OR c.name LIKE ?)"
            params.extend([f"%{category}%", f"%{category}%"])
        params.append(limit)
        cursor.execute(
            f"""
            SELECT p.*, c.name_fa as category_name_fa, c.name as category_name
            FROM products p
            JOIN categories c ON p.category_id = c.id
            JOIN products_fts ON products_fts.rowid = p.id
            WHERE products_fts MATCH ? {category_clause}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        )
        return cursor.fetchall()
    except sqlite3.OperationalError as exc:
        logger.debug("FTS search failed for %r: %s", query, exc)
        return []
    finally:
        conn.close()


def _search_products_like(query: str, category: Optional[str], limit: int) -> List[sqlite3.Row]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", query, flags=re.UNICODE)
    tokens = [query] if not tokens else tokens[:8]
    like_clauses = []
    params: List[Any] = []
    for token in tokens:
        like_clauses.append(
            "(p.name LIKE ? OR p.name_fa LIKE ? OR p.description LIKE ? OR p.description_fa LIKE ? OR p.specifications LIKE ?)"
        )
        params.extend([f"%{token}%"] * 5)

    category_clause = ""
    if category:
        category_clause = "AND (c.name_fa LIKE ? OR c.name LIKE ?)"
        params.extend([f"%{category}%", f"%{category}%"])

    params.append(limit)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT p.*, c.name_fa as category_name_fa, c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE ({" OR ".join(like_clauses)}) {category_clause}
        ORDER BY p.name_fa
        LIMIT ?
        """,
        params,
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_products_by_category(category_id: int) -> List[Dict[str, Any]]:
    """Get all products in a category."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.*, c.name_fa as category_name_fa, c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE p.category_id = ?
        ORDER BY p.name_fa
        """,
        (category_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_product(row) for row in rows]


def get_all_categories() -> List[Dict[str, Any]]:
    """Get all product categories."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM categories ORDER BY name_fa")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """Get a single product by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.*, c.name_fa as category_name_fa, c.name as category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE p.id = ?
        """,
        (product_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_product(row) if row else None


def _row_to_product(row: sqlite3.Row) -> Dict[str, Any]:
    product = dict(row)
    if product.get("specifications"):
        product["specifications"] = json.loads(product["specifications"])
    product["in_stock"] = bool(product.get("in_stock"))
    return product


def save_feedback(session_id: str, question: str, answer: str, rating: int, comment: str = "") -> bool:
    """Save user feedback for a response."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO feedback (session_id, question, answer, rating, comment) VALUES (?, ?, ?, ?, ?)",
            (session_id, question, answer, rating, comment),
        )
        conn.commit()
        return True
    except Exception as exc:
        logger.error("Failed to save feedback: %s", exc, exc_info=True)
        return False
    finally:
        conn.close()
