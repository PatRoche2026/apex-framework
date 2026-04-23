FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Phase 2: chromadb + voyageai (lightweight embeddings API)
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
