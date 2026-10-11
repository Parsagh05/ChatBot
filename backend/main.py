# backend/main.py
"""FastAPI backend for the RAG Chatbot with product integration and admin dashboard API."""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import admin
from chatbot import ERROR_REPLY, RAGChatbot
from config import settings
from knowledge_store import KnowledgeIndex
from product_db import (
    init_db,
    search_products,
    get_all_categories,
    get_products_by_category,
    get_product_by_id,
    save_feedback,
)
from security import chat_rate_limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

init_db()
chatbot = RAGChatbot(KnowledgeIndex())
admin.bind(chatbot)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # the index loads in the background so the server answers immediately
    admin.start_startup_job()
    yield


app = FastAPI(
    title="Smart Support Chatbot API",
    description="RAG-based chatbot with product database integration for customer support",
    version="5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)


# ─── Request/Response Models ────────────────────────────────────────────────


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    products: List[Dict[str, Any]] = []
    sources: List[Dict[str, str]] = []
    intent: str = "general_support"
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., max_length=100)
    question: str = Field(..., max_length=8000)
    answer: str = Field(..., max_length=8000)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(default="", max_length=2000)


def _session_id(requested: Optional[str]) -> str:
    if requested and requested.startswith("session_") and len(requested) < 60:
        return requested
    return f"session_{uuid.uuid4().hex}"


# ─── Chat Endpoints ─────────────────────────────────────────────────────────
# Plain `def` endpoints run in FastAPI's threadpool, so slow model calls don't block other users.


@app.post("/ask", response_model=QueryResponse, dependencies=[Depends(chat_rate_limit)])
def ask_question(request: QueryRequest):
    """Main chat endpoint; returns the full answer at once."""
    try:
        history = [turn.model_dump() for turn in request.history]
        result = chatbot.answer(request.question, history)
        return QueryResponse(
            answer=result["reply"],
            products=result["products"],
            sources=result["sources"],
            intent=result["intent"],
            session_id=_session_id(request.session_id),
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="خطا در پردازش سوال. لطفاً دوباره تلاش کنید.")


@app.post("/ask/stream", dependencies=[Depends(chat_rate_limit)])
def ask_question_stream(request: QueryRequest):
    """Chat endpoint streaming Server-Sent Events: meta, delta..., done."""
    session_id = _session_id(request.session_id)
    history = [turn.model_dump() for turn in request.history]

    def events():
        try:
            for event in chatbot.stream_answer(request.question, history):
                if event["type"] == "meta":
                    event["session_id"] = session_id
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error streaming answer: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'done', 'answer': ERROR_REPLY}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Product Endpoints ───────────────────────────────────────────────────────


@app.get("/products/search")
def search_products_endpoint(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    category: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Search products by keyword."""
    try:
        results = search_products(q, category=category, limit=limit)
        return {"products": results, "total": len(results)}
    except Exception as e:
        logger.error(f"Product search error: {e}")
        raise HTTPException(status_code=500, detail="خطا در جستجوی محصولات")


@app.get("/products/categories")
def get_categories():
    """Get all product categories."""
    categories = get_all_categories()
    return {"categories": categories}


@app.get("/products/category/{category_id}")
def get_products_in_category(category_id: int):
    """Get all products in a specific category."""
    products = get_products_by_category(category_id)
    return {"products": products, "total": len(products)}


@app.get("/products/{product_id}")
def get_product(product_id: int):
    """Get a single product by ID."""
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="محصول یافت نشد")
    return product


# ─── Utility Endpoints ───────────────────────────────────────────────────────


@app.get("/suggestions")
def get_suggestions():
    """Get suggested questions for new users."""
    return {"suggestions": chatbot.get_suggested_questions()}


@app.post("/feedback", dependencies=[Depends(chat_rate_limit)])
def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a chatbot response."""
    success = save_feedback(
        session_id=request.session_id,
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        comment=request.comment or "",
    )
    if success:
        return {"status": "ok", "message": "بازخورد شما ثبت شد. متشکریم!"}
    raise HTTPException(status_code=500, detail="خطا در ثبت بازخورد")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "5.0",
        "index_ready": chatbot.index.ready,
        "llm_ready": chatbot.llm is not None,
        "product_db": True,
    }
