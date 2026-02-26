# ==========================================
# UmaEdge — API Backend Dockerfile
# ==========================================

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY api/ api/
COPY models/ models/
COPY scraper/ scraper/
COPY notifications/ notifications/
COPY data/ data/

# Expose port
ENV PORT=8000
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/api/health', timeout=5)"

# Run with gunicorn for production
CMD ["sh", "-c", "gunicorn api.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT}"]
