#!/bin/bash
set -e

DATA_DIR="data/processed"
INDEX_FILE="${DATA_DIR}/vector_index.faiss"

echo "=================================================="
echo "Starting OSRS RAG Application Container..."
echo "=================================================="

# Ensure the local data target directory exists
mkdir -p "$DATA_DIR"

# Count any dataset/metadata/array files in data/processed
DATA_FILE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.npy" -o -name "*.faiss" \) 2>/dev/null | wc -l)

if [ ! -f "$INDEX_FILE" ] || [ "$DATA_FILE_COUNT" -eq 0 ]; then
    echo "Vector index or dataset files not found locally (Found $DATA_FILE_COUNT dataset files)."
    echo "Hugging Face dataset download will automatically trigger on startup in retriever.py."
    echo "=================================================="
else
    echo "Found valid existing vector index and dataset files in $DATA_DIR."
    echo "Skipping redundant downloads."
fi

echo "Starting application process..."
exec "$@"