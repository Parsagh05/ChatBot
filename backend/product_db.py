"""Product database management using SQLite, filled from the website snapshot."""

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


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


def import_products(products: List[Dict[str, Any]], force: bool = False) -> int:
    """Replace the product tables with the products from the website snapshot."""
    raw_json = json.dumps(products, ensure_ascii=False, sort_keys=True)
    source_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT source_hash FROM import_state WHERE source = ?", ("site_products",))
    row = cursor.fetchone()
    if row and row["source_hash"] == source_hash and not force:
        cursor.execute("SELECT COUNT(*) FROM products")
        count = int(cursor.fetchone()[0])
        conn.close()
        return count

    cursor.execute("DELETE FROM products")
    cursor.execute("DELETE FROM categories")
    cursor.execute("DELETE FROM products_fts")

    category_ids: Dict[str, int] = {}
    for product in products:
        category_name = (product.get("categories") or ["محصولات"])[0]
        if category_name not in category_ids:
            cursor.execute(
                """
                INSERT INTO categories (name, name_fa, description, description_fa)
                VALUES (?, ?, ?, ?)
                """,
                (category_name, category_name, None, None),
            )
            category_ids[category_name] = int(cursor.lastrowid)

        description = "\n".join(part for part in (product.get("short_description"), product.get("description")) if part)
        cursor.execute(
            """
            INSERT INTO products (
                source_chunk_id, source_url, name, name_fa, category_id,
                description, description_fa, price, currency, in_stock, specifications, image_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"site:{product['id']}",
                product.get("url", ""),
                _latin_name(product["name"]) or product["name"],
                product["name"],
                category_ids[category_name],
                description,
                description,
                product.get("price"),
                product.get("currency") or "تومان",
                1 if product.get("in_stock", True) else 0,
                json.dumps(_extract_specifications(product, category_name), ensure_ascii=False),
                product.get("image_url"),
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
        ("site_products", source_hash),
    )

    conn.commit()
    conn.close()
    logger.info("Imported %s products from the website snapshot", len(products))
    return len(products)


def _latin_name(title: str) -> str:
    parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9.+-]*)*", title)
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_specifications(product: Dict[str, Any], category: str) -> Dict[str, str]:
    specs: Dict[str, str] = {"عنوان": product["name"], "دسته‌بندی": category}
    specs.update(product.get("attributes") or {})

    # descriptions on the site are written as "key : value" lines
    for line in (product.get("description") or "").splitlines():
        line = line.strip().lstrip("-•").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key and value and len(key) <= 40:
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


def count_products() -> int:
    conn = get_connection()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])
    finally:
        conn.close()


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


def list_feedback(rating: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Newest feedback first. rating: None (all), 'down' (1-2) or 'up' (4-5)."""
    where = {"down": "WHERE rating <= 2", "up": "WHERE rating >= 4"}.get(rating or "", "")
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT id, session_id, question, answer, rating, comment, created_at FROM feedback {where} "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        stats = conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(rating >= 4), 0) AS up, COALESCE(SUM(rating <= 2), 0) AS down FROM feedback"
        ).fetchone()
        matching = conn.execute(f"SELECT COUNT(*) FROM feedback {where}").fetchone()[0]
    finally:
        conn.close()
    return {"items": [dict(row) for row in rows], "matching": int(matching), "stats": dict(stats)}


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
