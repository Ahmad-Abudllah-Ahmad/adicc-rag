# ADICC Volume 4 RAG API — Render / Docker
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/adicc.db ./data/adicc.db
COPY data/previews ./data/previews

ENV ADICC_DB=/app/data/adicc.db
ENV ADICC_PREVIEWS=/app/data/previews
ENV PYTHONUNBUFFERED=1

# Render sets PORT; default 8001 for local docker runs
EXPOSE 8001
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
