# backend/main.py
"""Enhanced FastAPI backend for the RAG Chatbot with product integration."""

import logging
import uuid
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from chatbot import RAGChatbot
from product_db import (
    init_db,
    import_products_from_knowledge_base,
    search_products,
    get_all_categories,
    get_products_by_category,
    get_product_by_id,
    save_feedback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize product database
init_db()
product_count = import_products_from_knowledge_base()
logger.info("Product database initialized from knowledge_base.json with %s products.", product_count)

# Initialize chatbot
chatbot = RAGChatbot()
logger.info("RAG Chatbot initialized successfully.")

app = FastAPI(
    title="Smart Support Chatbot API",
    description="RAG-based chatbot with product database integration for customer support",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    products: List[Dict[str, Any]] = []
    intent: str = "general_support"
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = ""


class ProductResponse(BaseModel):
    id: int
    name: str
    name_fa: str
    category_name_fa: str
    description_fa: str
    price: Optional[float]
    currency: str = "تومان"
    in_stock: bool
    specifications: Optional[Dict[str, str]] = None


# ─── Chat Endpoint ──────────────────────────────────────────────────────────


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Main chat endpoint with intent classification and product integration."""
    try:
        # Always generate server-side session ID if not provided or invalid
        if request.session_id and request.session_id.startswith("session_") and len(request.session_id) < 60:
            session_id = request.session_id
        else:
            session_id = f"session_{uuid.uuid4().hex}"

        # Step 1: Classify intent
        intent = chatbot.classify_intent(request.question)
        intent_type = intent.get("intent", "general_support")

        # Step 2: Rewrite query with history context
        standalone = chatbot.rewrite_query(request.question, request.history)

        # Step 3: Expand/correct query
        queries = chatbot.expand_query(standalone)

        # Step 4: Retrieve knowledge base chunks
        chunks = chatbot.retrieve_relevant_chunks(queries)

        # Step 5: Retrieve product information if relevant
        products = []
        product_context = ""
        if intent_type in ("product_search", "product_compare", "general_support"):
            products = chatbot.retrieve_product_info(standalone, intent)
            product_context = chatbot.format_product_context(products)

        # Step 6: Generate response with all context
        answer = chatbot.generate_response(
            request.question, chunks, request.history, product_context, intent
        )

        return QueryResponse(
            answer=answer,
            products=products,
            intent=intent_type,
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="خطا در پردازش سوال. لطفاً دوباره تلاش کنید.")


# ─── Product Endpoints ───────────────────────────────────────────────────────


@app.get("/products/search")
async def search_products_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    category: Optional[str] = None,
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
async def get_categories():
    """Get all product categories."""
    categories = get_all_categories()
    return {"categories": categories}


@app.get("/products/category/{category_id}")
async def get_products_in_category(category_id: int):
    """Get all products in a specific category."""
    products = get_products_by_category(category_id)
    return {"products": products, "total": len(products)}


@app.get("/products/{product_id}")
async def get_product(product_id: int):
    """Get a single product by ID."""
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="محصول یافت نشد")
    return product


# ─── Utility Endpoints ───────────────────────────────────────────────────────


@app.get("/suggestions")
async def get_suggestions():
    """Get suggested questions for new users."""
    return {"suggestions": chatbot.get_suggested_questions()}


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
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
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0",
        "chroma_available": chatbot.collection is not None,
        "product_db": True,
    }

