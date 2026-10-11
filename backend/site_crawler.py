"""Fetch the company website (WordPress + WooCommerce) into a local snapshot.

Uses the site's public JSON APIs rather than scraping HTML, so the text is the
page content without menus/footers and every URL is the site's exact permalink.
"""

import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "TabEnergyChatbot/1.0 (+knowledge-base sync)"
_DROP_TAGS = ["script", "style", "noscript", "svg", "form", "iframe", "button", "input", "select", "textarea"]


def snapshot_path() -> Path:
    return Path(settings.DATA_DIR) / "site_snapshot.json"


def load_snapshot() -> Optional[Dict[str, Any]]:
    path = snapshot_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Site snapshot %s is unreadable: %s", path, exc)
        return None


def html_to_text(markup: str) -> str:
    """Readable text from rendered WordPress HTML, one block per line."""
    if not markup:
        return ""
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    lines = []
    for line in soup.get_text("\n").splitlines():
        line = re.sub(r"[ \t ]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _plain(markup: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup or ""))).strip()


class _Client:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )

    def get_all(self, path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """GET every page of a paginated WordPress collection."""
        items: List[Dict[str, Any]] = []
        page = 1
        while True:
            response = self._get(path, {**params, "per_page": 100, "page": page})
            batch = response.json()
            if not isinstance(batch, list):
                raise ValueError(f"Unexpected response from {path}: {str(batch)[:200]}")
            items.extend(batch)
            total_pages = int(response.headers.get("X-WP-TotalPages") or 1)
            if page >= total_pages or not batch:
                return items
            page += 1

    def _get(self, path: str, params: Dict[str, Any]) -> httpx.Response:
        url = f"{self.base}{path}"
        last_error: Optional[Exception] = None
        for _ in range(3):
            try:
                response = self.http.get(url, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")

    def close(self) -> None:
        self.http.close()


def crawl_site(progress: Callable[[str], None] = lambda message: None) -> Dict[str, Any]:
    """Download pages, posts and products from the site and save the snapshot."""
    base_url = settings.SITE_URL
    client = _Client(base_url)
    try:
        progress("در حال دریافت دسته‌بندی‌ها...")
        categories = {
            c["id"]: html.unescape(c["name"])
            for c in client.get_all("/wp-json/wp/v2/categories", {"_fields": "id,name"})
        }

        progress("در حال دریافت صفحات سایت...")
        pages = client.get_all("/wp-json/wp/v2/pages", {"_fields": "id,link,title,content,modified"})

        progress("در حال دریافت مقالات، خدمات و پروژه‌ها...")
        posts = client.get_all("/wp-json/wp/v2/posts", {"_fields": "id,link,title,content,categories,date,modified"})

        progress("در حال دریافت محصولات...")
        products = _fetch_products(client)
    finally:
        client.close()

    documents: List[Dict[str, Any]] = []
    for page in pages:
        text = html_to_text(page["content"]["rendered"])
        if len(text) < 40:  # cart, checkout, empty form pages
            continue
        documents.append(
            {
                "id": f"page:{page['id']}",
                "source_type": "page",
                "title": _plain(page["title"]["rendered"]),
                "url": page["link"],
                "categories": [],
                "text": text,
                "modified": page.get("modified", ""),
            }
        )

    for post in posts:
        text = html_to_text(post["content"]["rendered"])
        if not text:
            continue
        documents.append(
            {
                "id": f"post:{post['id']}",
                "source_type": "post",
                "title": _plain(post["title"]["rendered"]),
                "url": post["link"],
                "categories": [categories[c] for c in post.get("categories", []) if c in categories],
                "text": text,
                "modified": post.get("modified", ""),
            }
        )

    snapshot = {
        "site_url": base_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": documents,
        "products": products,
    }
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    logger.info("Site snapshot saved: %s documents, %s products", len(documents), len(products))
    return snapshot


def _fetch_products(client: _Client) -> List[Dict[str, Any]]:
    """Merge WooCommerce store data (price, stock, images) with WordPress product text."""
    store_items = client.get_all("/wp-json/wc/store/v1/products", {})
    wp_items = {
        item["id"]: item
        for item in client.get_all("/wp-json/wp/v2/product", {"_fields": "id,link,title,content,excerpt,modified"})
    }

    products = []
    for item in store_items:
        wp = wp_items.get(item["id"], {})
        description = html_to_text(item.get("description") or "") or html_to_text(wp.get("content", {}).get("rendered", ""))
        short = html_to_text(item.get("short_description") or "") or html_to_text(wp.get("excerpt", {}).get("rendered", ""))
        prices = item.get("prices") or {}
        minor = int(prices.get("currency_minor_unit") or 0)
        price = int(prices.get("price") or 0) / (10 ** minor)
        images = item.get("images") or []
        products.append(
            {
                "id": item["id"],
                "name": html.unescape(item.get("name") or ""),
                "url": item.get("permalink") or wp.get("link", ""),
                "categories": [html.unescape(c["name"]) for c in item.get("categories", [])],
                "short_description": short,
                "description": description,
                "price": price or None,
                "currency": (prices.get("currency_suffix") or " تومان").strip(),
                "in_stock": bool(item.get("is_in_stock", True)),
                "image_url": images[0]["src"] if images else None,
                "attributes": {
                    html.unescape(a.get("name", "")): ", ".join(html.unescape(t.get("name", "")) for t in a.get("terms", []))
                    for a in item.get("attributes", [])
                    if a.get("name")
                },
            }
        )
    return products
