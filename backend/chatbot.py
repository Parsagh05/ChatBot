# backend/chatbot.py
"""Enhanced RAG Chatbot with product database integration."""

import json
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any, Optional

from config import settings
from llm_client import LLMClient
from product_db import search_products
from knowledge_store import ensure_knowledge_base_index, load_suggested_questions

logger = logging.getLogger(__name__)


class RAGChatbot:
    """RAG-based chatbot with product knowledge integration."""

    def __init__(self, api_key: str = None, base_url: str = None):
        # Embedding model
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        self.model = settings.LLM_MODEL
        self.llm = LLMClient(
            model=self.model,
            openai_api_key=api_key or settings.OPENAI_API_KEY,
            openai_base_url=base_url or settings.OPENAI_BASE_URL,
            gemini_api_key=settings.GEMINI_API_KEY,
            gemini_api_endpoint=settings.GEMINI_API_ENDPOINT,
        )
        logger.info("LLM provider selected: %s for model %s", self.llm.provider, self.model)

        # ChromaDB index, built from knowledge_base/*.json when needed.
        self.collection = None
        try:
            self.collection = ensure_knowledge_base_index(self.embedding_model)
            logger.info("ChromaDB collection loaded successfully from JSON knowledge base.")
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}. Product-only mode active.")

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify user intent to route to appropriate handler."""
        classify_prompt = f"""
شما یک طبقه‌بندی‌کننده هوشمند هستید. سوال کاربر را تحلیل کنید و مشخص کنید:
1. intent: نوع سوال (product_search, product_compare, general_support, greeting, farewell)
2. product_keywords: کلمات کلیدی مرتبط با محصول (اگر وجود دارد)
3. category_hint: دسته‌بندی محصول (اگر مشخص است)

پاسخ را فقط به صورت JSON بدون هیچ توضیح اضافه برگردانید.

سوال: "{query}"
"""
        try:
            response_text = self.llm.chat(
                messages=[
                    {"role": "system", "content": "شما یک طبقه‌بندی‌کننده سوالات هستید. فقط JSON خالص برگردانید."},
                    {"role": "user", "content": classify_prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            response_text = response_text.strip()
            # Strip markdown code fences if present (only at boundaries)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)
            return json.loads(response_text)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Intent classification failed: {e}")
            return {"intent": "general_support", "product_keywords": [], "category_hint": None}

    def rewrite_query(self, current_query: str, history: List[Dict[str, str]]) -> str:
        """Rewrite query to be standalone using conversation history."""
        if not history:
            return current_query

        history_str = "\n".join(
            [
                f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
                for turn in history[-6:]
            ]
        )

        rewrite_prompt = f"""
شما یک دستیار هوشمند هستید که وظیفه دارد سوال فعلی کاربر را با توجه به تاریخچه گفتگو، به یک سوال کاملاً مستقل و واضح تبدیل کند.
- اگر سوال فعلی کامل و مستقل است، همان را بدون تغییر برگردانید.
- در غیر این صورت، آن را طوری بازنویسی کنید که بدون نیاز به تاریخچه قابل فهم باشد.
- فقط سوال بازنویسی‌شده را در یک خط و بدون هیچ توضیح اضافه بنویسید.

تاریخچه گفتگو:
{history_str}

سوال فعلی: "{current_query}"
"""

        try:
            return self.llm.chat(
                messages=[
                    {"role": "system", "content": "شما یک بازنویس‌کننده هوشمند سوالات فارسی هستید."},
                    {"role": "user", "content": rewrite_prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            ).strip()
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return current_query

    def expand_query(self, query: str) -> List[str]:
        """Correct spelling and expand query for better retrieval."""
        correction_prompt = f"""
شما یک ویرایشگر زبان فارسی هستید. وظیفه شما این است که:
1. اشتباهات املایی، نگارشی یا دستوری سوال زیر را اصلاح کنید.
2. اگر به صورت رسمی نوشته نشده آن را به صورت رسمی بازنویسی کنید.
3. اگر سوال از نظر زبانی صحیح است، همان سوال را بدون هیچ تغییری برگردانید.
- فقط یک خط پاسخ بدهید و هیچ توضیح اضافه ننویسید.

سوال: "{query}"
"""

        try:
            corrected = self.llm.chat(
                messages=[
                    {"role": "system", "content": "شما یک ویرایشگر دقیق زبان فارسی هستید."},
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0.0,
            ).strip()
            if not corrected:
                corrected = query
            queries = [query]
            if corrected != query:
                queries.append(corrected)
            return queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    def retrieve_relevant_chunks(self, queries: List[str], n_results: int = 6) -> List[Tuple[str, Dict[str, Any]]]:
        """Retrieve relevant document chunks from ChromaDB."""
        if not self.collection:
            return []

        seen = set()
        results = []
        for q in queries:
            try:
                emb = self.embedding_model.encode(q).tolist()
                res = self.collection.query(query_embeddings=[emb], n_results=n_results)
                for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                    if doc not in seen:
                        seen.add(doc)
                        results.append((doc, meta))
            except Exception as e:
                logger.error(f"Retrieval error for query '{q}': {e}")

        return results

    def retrieve_product_info(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search the product database based on user query and intent."""
        keywords = intent.get("product_keywords", [])
        category_hint = intent.get("category_hint")

        # Build search terms
        search_terms = []
        if keywords:
            search_terms.extend(keywords)
        else:
            # Use the query itself as search
            search_terms.append(query)

        products = []
        seen_ids = set()
        for term in search_terms:
            try:
                results = search_products(term, category=category_hint, limit=5)
                for p in results:
                    if p["id"] not in seen_ids:
                        seen_ids.add(p["id"])
                        products.append(p)
            except Exception as e:
                logger.error(f"Product search error for '{term}': {e}")

        return products[:5]

    def format_product_context(self, products: List[Dict[str, Any]]) -> str:
        """Format product information for LLM context."""
        if not products:
            return ""

        parts = []
        for i, p in enumerate(products, 1):
            specs_str = ""
            if p.get("specifications") and isinstance(p["specifications"], dict):
                specs_str = " | ".join([f"{k}: {v}" for k, v in p["specifications"].items()])

            stock_status = "موجود ✓" if p.get("in_stock") else "ناموجود ✗"
            price_str = f"{p['price']:,.0f} {p.get('currency', 'تومان')}" if p.get("price") else "قیمت: تماس بگیرید"

            part = f"""محصول {i}: {p['name_fa']} ({p['name']})
دسته‌بندی: {p.get('category_name_fa', 'نامشخص')}
توضیحات: {p.get('description_fa', '')}
قیمت: {price_str}
وضعیت: {stock_status}
مشخصات: {specs_str}"""
            parts.append(part)

        return "\n\n---\n\n".join(parts)

    def generate_response(
        self,
        query: str,
        context_chunks: List[Tuple[str, Dict]],
        history: List[Dict],
        product_context: str = "",
        intent: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate final response using LLM with all available context."""
        if not self.llm:
            return "سیستم در دسترس نیست. لطفاً بعداً تلاش کنید."

        # Handle greetings/farewells without needing context
        if intent and intent.get("intent") == "greeting":
            return "سلام! 👋 من پشتیبان هوشمند شما هستم. چطور می‌تونم کمکتون کنم؟ می‌تونید درباره محصولات ما سوال بپرسید یا هر سوال دیگه‌ای دارید بفرمایید."

        if intent and intent.get("intent") == "farewell":
            return "خداحافظ! 🙏 امیدوارم تونسته باشم کمکتون کنم. هر وقت سوالی داشتید، در خدمتم."

        # Build context sections
        context_parts = []
        for i, (text, meta) in enumerate(context_chunks, 1):
            source_title = meta.get("source_title", "منبع نامشخص")
            parent_cat = meta.get("parent_category")
            ref = f"منبع {i}: بخش «{source_title}»"
            if parent_cat:
                ref += f" (زیرمجموعهٔ «{parent_cat}»)"
            url = meta.get("source_url", "").strip()
            if url and url != "N/A":
                ref += f" — لینک: {url}"
            context_parts.append(f"{ref}\nمتن مرتبط: {text}")

        knowledge_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # Build history section
        history_section = ""
        if history:
            history_str = "\n".join(
                [
                    f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
                    for turn in history[-6:]
                ]
            )
            history_section = f"\n\nتاریخچه گفتگوی اخیر:\n{history_str}\n"

        # Combined system prompt
        system_prompt = """شما یک دستیار خدمات مشتری متخصص، دوستانه و حرفه‌ای هستید. نام شما «پشتیبان» است.

قوانین شما:
1. فقط بر اساس «زمینه» و «اطلاعات محصولات» ارائه شده پاسخ دهید.
2. اطلاعاتی از خودتان اضافه نکنید.
3. اگر اطلاعات کافی ندارید، مودبانه بگویید و پیشنهاد دهید با پشتیبانی تماس بگیرند.
4. اگر محصولی ناموجود است، این را بگویید و محصولات مشابه موجود را پیشنهاد دهید.
5. برای مقایسه محصولات، جدول مقایسه‌ای ارائه دهید.
6. قیمت‌ها را با فرمت خوانا نمایش دهید.
7. اگر URL ذکر شده، لینک مرجع ارائه دهید.
8. همیشه و فقط به زبان فارسی پاسخ دهید.
9. از ایموجی مناسب استفاده کنید تا پاسخ‌ها جذاب‌تر باشند.
10. پاسخ‌ها را ساختاریافته و با فرمت مناسب (بولت پوینت، عناوین) ارائه دهید."""

        # Build user prompt
        user_prompt_parts = []
        if knowledge_context:
            user_prompt_parts.append(f"زمینه از پایگاه دانش:\n{knowledge_context}")
        if product_context:
            user_prompt_parts.append(f"اطلاعات محصولات:\n{product_context}")
        if history_section:
            user_prompt_parts.append(history_section)
        user_prompt_parts.append(f"سوال فعلی: \"{query}\"")

        user_prompt = "\n\n".join(user_prompt_parts)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            return self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
            )
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "متأسفانه یک خطای غیرمنتظره رخ داد. لطفاً دوباره تلاش کنید."

    def get_suggested_questions(self) -> List[str]:
        """Return suggested questions from the JSON FAQ knowledge base."""
        questions = load_suggested_questions(limit=5)
        if questions:
            return questions
        return ["چه محصولاتی برای فروش دارید؟", "چطور می‌توانم با شما تماس بگیرم؟"]

