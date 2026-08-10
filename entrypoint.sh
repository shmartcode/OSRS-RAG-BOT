#!/usr/bin/env bash
set -e

PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
FAISS_INDEX="${PROCESSED_DIR}/vector_index.faiss"
METADATA_FILE="${PROCESSED_DIR}/metadata.jsonl"

echo "=========================================="
echo "Starting OSRS RAG Application Service"
echo "=========================================="

# 1. Ensure processed directory exists
mkdir -p "$PROCESSED_DIR"

# 2. Check for core runtime vector search artifacts
if [ ! -f "$FAISS_INDEX" ] || [ ! -f "$METADATA_FILE" ]; then
    echo "[INFO] Vector artifacts missing from local storage (${FAISS_INDEX} / ${METADATA_FILE})."
    echo "[INFO] Retriever will download runtime assets from Hugging Face Hub on initialization."
else
    echo "[OK] Found local FAISS index: ${FAISS_INDEX}"
    echo "[OK] Found local metadata: ${METADATA_FILE}"
fi

# 3. Optional: Clean up legacy .npy or separate .json metadata files if present
echo "[INFO] Cleaning up legacy embedding files if present..."
rm -f "${PROCESSED_DIR}"/*_embeddings.npy
rm -f "${PROCESSED_DIR}"/*_metadata.json

# 4. Start the application (Uvicorn / FastAPI server)
echo "Starting application process..."
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec uvicorn src.api:app --host 0.0.0.0 --port 8000
fi