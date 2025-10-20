# backend/chatbot.py
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os
from typing import List, Tuple, Dict, Any

class RAGChatbot:
    def __init__(self, api_key: str, base_url: str = "https://api.tapsage.com/openai/v1"):
        # Embedding model
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        
        # OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini"

        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="../chroma_db")
        self.collection = self.chroma_client.get_collection(name="farsi_rag_collection")

    def rewrite_query(self, current_query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return current_query

        history_str = "\n".join([
            f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
            for turn in history[-4:]
        ])

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
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک بازنویس‌کننده هوشمند سوالات فارسی هستید."},
                    {"role": "user", "content": rewrite_prompt}
                ],
                temperature=0.0,
                max_tokens=60
            )
            return completion.choices[0].message.content.strip()
        except:
            return current_query

    def expand_query(self, query: str) -> List[str]:
        correction_prompt = f"""
        شما یک ویرایشگر زبان فارسی هستید. وظیفه شما این است که فقط و فقط اشتباهات املایی، نگارشی یا دستوری سوال زیر را اصلاح کنید.
        - اگر به صورت رسمی نوشته نشده آن را به صورت رسمی بازنویسی کنید.
        - اگر سوال از نظر زبانی صحیح است، همان سوال را بدون هیچ تغییری برگردانید.
        - هیچ بازنویسی معنادار، تغییر سبک یا جایگزینی واژه انجام ندهید.
        - فقط یک خط پاسخ بدهید و هیچ توضیح اضافه ننویسید.

        سوال: "{query}"
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "شما یک ویرایشگر دقیق زبان فارسی هستید."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.0,
            )
            corrected = completion.choices[0].message.content.strip()
            if not corrected:
                corrected = query
            return [query, corrected]
        except:
            return [query]

    def retrieve_relevant_chunks(self, queries: List[str], n_results: int = 6) -> List[Tuple[str, Dict[str, Any]]]:
        seen = set()
        results = []
        for q in queries:
            emb = self.embedding_model.encode(q).tolist()
            res = self.collection.query(query_embeddings=[emb], n_results=n_results)
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                if doc not in seen:
                    seen.add(doc)
                    results.append((doc, meta))
        return results

    def generate_response(self, query: str, context_chunks: List[Tuple[str, Dict]], history: List[Dict]) -> str:
        """
        Generates a final response using the LLM based on the query and retrieved context (with metadata).
        """
        if not self.client:
            return "OpenAI client is not initialized. Cannot generate response."

        if not context_chunks:
            return "متاسفانه اطلاعات کافی برای پاسخ به سوال شما در پایگاه دانش من وجود ندارد."

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

        context_string = "\n\n---\n\n".join(context_parts)

        # Build conversation history for LLM (last few turns)
        history_str = ""
        if history:
            history_str = "\n".join([
                f"{'کاربر' if turn['role'] == 'user' else 'پشتیبان'}: {turn['content']}"
                for turn in history[-4:]
            ])
            history_section = f"\n\nتاریخچه گفتگوی اخیر:\n{history_str}\n"
        else:
            history_section = ""

        system_prompt = """
        شما یک دستیار خدمات مشتری متخصص و دوستانه برای یک شرکت هستید. نام شما «پشتیبان» است.
        شما باید فقط و فقط بر اساس «زمینه» ارائه شده به «سوال» کاربر پاسخ دهید.
        اطلاعاتی را از خودتان اضافه نکنید. اگر زمینه برای پاسخ دادن کافی نیست،
        مودبانه بگویید که اطلاعات کافی برای پاسخ به آن سوال را ندارید.
        اگر در زمینه **آدرس سایت (URL)** ذکر شده باشد، می‌توانید بگویید:
        «برای اطلاعات بیشتر می‌توانید به بخش (عنوان بخش) در سایت ما مراجعه کنید: [لینک]»
        ***شما باید همیشه و فقط به زبان فارسی پاسخ دهید.***
        """

        user_prompt = f"زمینه: \"{context_string}\"\nتاریخچه گفتمان: {history_section}\n\nسوال فعلی: \"{query}\""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
            )
            return completion.choices[0].message.content
        except:
            return "یک خطای غیرمنتظره رخ داد. لطفا دوباره تلاش کنید."