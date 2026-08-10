FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOCAL_LLM_URL="http://ollama:11434/v1"
ENV LOCAL_MODEL_NAME="llama3.1:8b"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download and cache the Hugging Face model during the image build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shmartcode/osrs-embedder-v2')"

# Grant execution permissions to entrypoint.sh
RUN chmod +x entrypoint.sh

# Set the entrypoint script and default start command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]