FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOCAL_LLM_URL="http://ollama:11434/v1"
ENV LOCAL_MODEL_NAME="llama3.1:8b"
ENV HF_HOME=/app/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download and cache the Hugging Face model and cross reranker during the image build
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
    SentenceTransformer('shmartcode/osrs-embedder-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Fix Windows line endings (converts \r\n to \n) and make it executable
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set it as the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["python", "app.py"]


