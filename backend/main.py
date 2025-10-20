# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from chatbot import RAGChatbot
from fastapi.middleware.cors import CORSMiddleware

# Initialize once (singleton)
API_KEY = "tpsg-HURAqpPOuLGZtEReVzj3unfTRXGp45o"  # 🔒 In production, use environment variable!
chatbot = RAGChatbot(api_key=API_KEY)

app = FastAPI(title="Farsi RAG Chatbot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # اجازه دادن به فرانت‌اند
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Step 1: Rewrite query
        standalone = chatbot.rewrite_query(request.question, request.history)
        
        # Step 2: Expand/correct
        queries = chatbot.expand_query(standalone)
        
        # Step 3: Retrieve
        chunks = chatbot.retrieve_relevant_chunks(queries)
        
        # Step 4: Generate
        answer = chatbot.generate_response(request.question, chunks, request.history)
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Optional: Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}