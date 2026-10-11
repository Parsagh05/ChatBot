# backend/chatbot.py
"""RAG chatbot: one analysis call, retrieval, then a (streamable) answer with exact source links."""

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from config import settings
from knowledge_store import KnowledgeIndex, load_suggested_questions
from links import display_url
from llm_client import LLMClient
from product_db import search_products

logger = logging.getLogger(__name__)

INTENTS = {"product_search", "product_compare", "general_support", "greeting", "farewell"}
MAX_SOURCES = 4

GREETING_REPLY = "سلام! 👋 من پشتیبان هوشمند شما هستم. چطور می‌تونم کمکتون کنم؟ می‌تونید درباره محصولات ما سوال بپرسید یا هر سوال دیگه‌ای دارید بفرمایید."
FAREWELL_REPLY = "خداحافظ! 🙏 امیدوارم تونسته باشم کمکتون کنم. هر وقت سوالی داشتید، در خدمتم."
LLM_UNAVAILABLE_REPLY = "سرویس پاسخ‌گویی هوشمند در حال حاضر پیکربندی نشده است. لطفاً بعداً تلاش کنید."
ERROR_REPLY = "متأسفانه یک خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید."
INDEX_LOADING_REPLY = "در حال آماده‌سازی پایگاه دانش هستم ⏳ لطفاً چند لحظه دیگر دوباره سوال خود را بپرسید."

SYSTEM_PROMPT = """شما یک دستیار خدمات مشتری متخصص، دوستانه و حرفه‌ای شرکت «تابان انرژی مازند سو» (TabEnergy) هستید. نام شما «پشتیبان» است.
در ابتدای هر پاسخ سلام نکنید و خودتان را معرفی نکنید؛ مستقیم به سوال پاسخ دهید.

قوانین شما:
1. فقط بر اساس «زمینه» و «اطلاعات محصولات» ارائه شده پاسخ دهید.
2. اطلاعاتی از خودتان اضافه نکنید.
3. اگر اطلاعات کافی ندارید، مودبانه بگویید و پیشنهاد دهید با پشتیبانی تماس بگیرند.
4. اگر محصولی ناموجود است، این را بگویید و محصولات مشابه موجود را پیشنهاد دهید.
5. برای مقایسه محصولات، جدول مقایسه‌ای ارائه دهید.
6. قیمت‌ها را با فرمت خوانا نمایش دهید.
7. برای ارجاع فقط از لینک‌هایی که در زمینه آمده استفاده کنید و آن‌ها را دقیقاً کپی کنید؛ هرگز لینک جدید نسازید.
8. همیشه و فقط به زبان فارسی پاسخ دهید.
9. از ایموجی مناسب استفاده کنید تا پاسخ‌ها جذاب‌تر باشند.
10. پاسخ‌ها را ساختاریافته و با فرمت مناسب (بولت پوینت، عناوین) ارائه دهید."""


def _history_text(history: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}" for turn in history[-6:]
    )


class RAGChatbot:
    """RAG-based chatbot with product knowledge integration."""

    def __init__(self, index: KnowledgeIndex):
        self.index = index
        self.llm: Optional[LLMClient] = None
        self.llm_error = ""
        self.reload_llm()

    def reload_llm(self) -> None:
        """(Re)create the LLM client from the current settings."""
        try:
            self.llm = LLMClient(
                model=settings.LLM_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_base_url=settings.OPENAI_BASE_URL,
                gemini_api_key=settings.GEMINI_API_KEY,
                gemini_api_endpoint=settings.GEMINI_API_ENDPOINT,
                provider=settings.LLM_PROVIDER,
            )
            self.llm_error = ""
            logger.info("LLM provider selected: %s for model %s", self.llm.provider, settings.LLM_MODEL)
        except Exception as exc:
            self.llm, self.llm_error = None, str(exc)
            logger.error("LLM client unavailable: %s", exc)

    # ─── Understanding ──────────────────────────────────────────────────────

    def analyze_query(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Intent, standalone rewrite and spelling correction in a single LLM call."""
        fallback = {
            "intent": "general_support",
            "standalone_query": query,
            "corrected_query": query,
            "product_keywords": [],
            "category_hint": None,
        }
        if not self.llm:
            return fallback

        prompt = f"""تاریخچه گفتگو (ممکن است خالی باشد):
{_history_text(history) or "-"}

پیام فعلی کاربر: "{query}"

فقط یک شیء JSON با این کلیدها برگردانید:
- "intent": یکی از product_search، product_compare، general_support، greeting، farewell. greeting و farewell فقط وقتی که پیام صرفاً سلام یا خداحافظی است و سوالی ندارد.
- "standalone_query": پیام فعلی به‌صورت یک سوال مستقل و کامل با توجه به تاریخچه (اگر مستقل است، همان را بنویسید). فقط ارجاع‌ها به تاریخچه را کامل کنید و هیچ کلمه‌ای که کاربر نگفته اضافه نکنید.
- "corrected_query": همان سوال مستقل با املای اصلاح‌شده و نگارش رسمی، بدون افزودن کلمه جدید.
- "product_keywords": فهرست کلمات کلیدی محصول (یا فهرست خالی).
- "category_hint": دسته‌بندی محصول یا null."""
        try:
            text = self.llm.chat(
                messages=[
                    {"role": "system", "content": "شما تحلیل‌گر سوالات کاربران یک شرکت انرژی خورشیدی هستید. فقط JSON معتبر برگردانید."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        except Exception as exc:
            logger.warning("Query analysis failed: %s", exc)
            return fallback

        result = dict(fallback)
        if data.get("intent") in INTENTS:
            result["intent"] = data["intent"]
        for key in ("standalone_query", "corrected_query"):
            if isinstance(data.get(key), str) and data[key].strip():
                result[key] = data[key].strip()
        if isinstance(data.get("product_keywords"), list):
            result["product_keywords"] = [str(k) for k in data["product_keywords"] if str(k).strip()][:5]
        if isinstance(data.get("category_hint"), str) and data["category_hint"].strip():
            result["category_hint"] = data["category_hint"].strip()
        return result

    # ─── Retrieval ──────────────────────────────────────────────────────────

    def retrieve_product_info(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search the product database based on user query and intent."""
        search_terms = analysis.get("product_keywords") or [query]
        products = []
        seen_ids = set()
        for term in search_terms:
            try:
                for p in search_products(term, category=analysis.get("category_hint"), limit=5):
                    if p["id"] not in seen_ids:
                        seen_ids.add(p["id"])
                        products.append(p)
            except Exception as e:
                logger.error(f"Product search error for '{term}': {e}")
        return products[:5]

    def format_product_context(self, products: List[Dict[str, Any]]) -> str:
        """Format product information for LLM context."""
        parts = []
        for i, p in enumerate(products, 1):
            specs_str = ""
            if p.get("specifications") and isinstance(p["specifications"], dict):
                specs_str = " | ".join([f"{k}: {v}" for k, v in p["specifications"].items()])

            stock_status = "موجود ✓" if p.get("in_stock") else "ناموجود ✗"
            price_str = f"{p['price']:,.0f} {p.get('currency', 'تومان')}" if p.get("price") else "قیمت: تماس بگیرید"
            link = display_url(self.index.resolver.resolve(p.get("source_url") or ""))

            part = f"""محصول {i}: {p['name_fa']}
دسته‌بندی: {p.get('category_name_fa', 'نامشخص')}
توضیحات: {p.get('description_fa', '')}
قیمت: {price_str}
وضعیت: {stock_status}
مشخصات: {specs_str}"""
            if link:
                part += f"\nلینک محصول: {link}"
            parts.append(part)

        return "\n\n---\n\n".join(parts)

    def _sources(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
        sources, seen = [], set()
        for _, meta in chunks:
            url = meta.get("source_url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": meta.get("source_title") or display_url(url), "url": url, "display": display_url(url)})
        return sources[:MAX_SOURCES]

    # ─── Answering ──────────────────────────────────────────────────────────

    def prepare(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Everything up to the final generation call."""
        analysis = self.analyze_query(query, history)
        intent = analysis["intent"]
        prepared: Dict[str, Any] = {"intent": intent, "products": [], "sources": [], "reply": None, "messages": None}

        if intent == "greeting":
            prepared["reply"] = GREETING_REPLY
            return prepared
        if intent == "farewell":
            prepared["reply"] = FAREWELL_REPLY
            return prepared
        if not self.llm:
            prepared["reply"] = LLM_UNAVAILABLE_REPLY
            return prepared
        if not self.index.ready:
            prepared["reply"] = INDEX_LOADING_REPLY
            return prepared

        # the user's own wording is always searched too, so a bad rewrite cannot hide the right sources
        queries = list(dict.fromkeys([query, analysis["standalone_query"], analysis["corrected_query"]]))
        try:
            chunks = self.index.search(queries, settings.TOP_K)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            chunks = []
        products = self.retrieve_product_info(analysis["standalone_query"], analysis)

        context_parts = []
        for i, (text, meta) in enumerate(chunks, 1):
            ref = f"منبع {i}: «{meta.get('source_title', 'منبع نامشخص')}»"
            if meta.get("categories"):
                ref += f" (دسته: {meta['categories']})"
            if meta.get("source_url"):
                ref += f" — لینک: {display_url(meta['source_url'])}"
            context_parts.append(f"{ref}\nمتن مرتبط: {text}")

        user_prompt_parts = []
        if context_parts:
            user_prompt_parts.append("زمینه از پایگاه دانش:\n" + "\n\n---\n\n".join(context_parts))
        if products:
            user_prompt_parts.append("اطلاعات محصولات:\n" + self.format_product_context(products))
        if history:
            user_prompt_parts.append(f"تاریخچه گفتگوی اخیر:\n{_history_text(history)}")
        user_prompt_parts.append(f"سوال فعلی: \"{query}\"")

        prepared["products"] = products
        prepared["sources"] = self._sources(chunks)
        prepared["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_prompt_parts)},
        ]
        return prepared

    def finalize(self, text: str) -> str:
        """Point every link in the answer at the exact website URL."""
        return self.index.resolver.fix_links(text)

    def answer(self, query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        prepared = self.prepare(query, history)
        if prepared["reply"] is None:
            try:
                text = self.llm.chat(prepared["messages"], temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS)
                prepared["reply"] = self.finalize(text)
            except Exception as exc:
                logger.error("Response generation failed: %s", exc)
                prepared["reply"] = ERROR_REPLY
        return prepared

    def stream_answer(self, query: str, history: List[Dict[str, str]]) -> Iterator[Dict[str, Any]]:
        """Events: one 'meta', any number of 'delta', then 'done' with the link-fixed full answer."""
        prepared = self.prepare(query, history)
        yield {"type": "meta", "intent": prepared["intent"], "products": prepared["products"], "sources": prepared["sources"]}

        if prepared["reply"] is not None:
            yield {"type": "done", "answer": prepared["reply"]}
            return

        pieces: List[str] = []
        try:
            for piece in self.llm.stream_chat(prepared["messages"], temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS):
                pieces.append(piece)
                yield {"type": "delta", "text": piece}
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            if not pieces:
                yield {"type": "done", "answer": ERROR_REPLY}
                return
        yield {"type": "done", "answer": self.finalize("".join(pieces))}

    def get_suggested_questions(self) -> List[str]:
        """Return suggested questions from the JSON FAQ knowledge base."""
        questions = load_suggested_questions(limit=5)
        if questions:
            return questions
        return ["چه محصولاتی برای فروش دارید؟", "چطور می‌توانم با شما تماس بگیرم؟"]
