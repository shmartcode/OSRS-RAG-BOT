#!/bin/bash
set -e

# Path to the primary deliverable file
INDEX_FILE="data/processed/vector_index.faiss"

echo "=================================================="
echo "Starting OSRS RAG Application Container..."
echo "=================================================="

# Check if the FAISS index exists in the processed directory
if [ ! -f "$INDEX_FILE" ]; then
    echo "FAISS Index not found at $INDEX_FILE."
    echo "Running full data ingestion and embedding pipeline..."
    echo "=================================================="

    # Step 1: Fetch raw data (Wiki API, GE Prices) Ge data is fetched and parsed in ge_data_fetcher.py
    echo "[1/4] Fetching and parsing raw data..."
    if [ -f "src/data_handling/ingestion/wiki_data_fetcher.py" ]; then
        python src/data_handling/ingestion/wiki_data_fetcher.py
    else
        echo "No separate wiki data fetcher script found.."
    fi
    if [ -f "src/data_handling/ingestiong/ge_data_fetcher.py" ]; then
        python src/data_handling/ingestion/ge_data_fetcher.py
    else
        echo "No separate GE data fetcher script found.."
    fi

    # Step 2: Parse raw dumps into structured JSONL (osrsreboxed fetches and parses in reboxed_data_parser.py)
    python src/data_handling/ingestion/wiki_data_parser.py
    python src/data_handling/reboxed_data_parser.py
    
    # Step 3: Format text blocks & generate embeddings
    echo "[3/4] Generating embeddings with fine-tuned model..."
    python src/data_handling/format_embed.py

    # Step 4: Build FAISS vector index
    echo "[4/4] Constructing FAISS vector index..."
    python src/data_handling/build_vector_index.py

    echo "=================================================="
    echo "Data pipeline execution complete!"
    echo "=================================================="
else
    echo "Found existing vector index at $INDEX_FILE."
    echo "Skipping data pipeline step."
fi

echo "Starting application process..."
# Execute whatever command was passed via Dockerfile CMD or Compose
exec "$@"