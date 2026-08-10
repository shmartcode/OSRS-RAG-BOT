#!/bin/bash
set -e

INDEX_FILE="data/processed/vector_index.faiss"

echo "=================================================="
echo "Starting OSRS RAG Application Container..."
echo "=================================================="

# Force run if the index file is missing OR if no metadata json files exist
METADATA_COUNT=$(find data/processed -name "*_metadata.json" 2>/dev/null | wc -l)

if [ ! -f "$INDEX_FILE" ] || [ "$METADATA_COUNT" -eq 0 ]; then
    echo "FAISS Index or metadata missing (Found $METADATA_COUNT metadata files)."
    echo "Running full data ingestion and embedding pipeline..."
    echo "=================================================="

    # Step 1: Fetch raw data
    echo "[1/4] Fetching and parsing raw data..."
    if [ -f "src/data_handling/ingestion/wiki_data_fetcher.py" ]; then
        python src/data_handling/ingestion/wiki_data_fetcher.py
    fi
    if [ -f "src/data_handling/ingestion/ge_data_fetcher.py" ]; then
        python src/data_handling/ingestion/ge_data_fetcher.py
    fi

    # Step 2: Parse raw dumps
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
    echo "Found valid existing vector index and metadata."
    echo "Skipping data pipeline step."
fi

echo "Starting application process..."
exec "$@"