FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    CHROMA_DB_PATH=/data/chroma_db \
    PRODUCT_DB_PATH=/data/products.db \
    DATA_DIR=/data/app \
    KNOWLEDGE_BASE_DIR=/knowledge_base

WORKDIR /app/backend

# CPU-only PyTorch keeps the image about 2 GB smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker: rate limits, admin sessions and background jobs live in process memory.
# --proxy-headers takes the client IP from nginx's X-Forwarded-For, so rate limits apply per visitor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
