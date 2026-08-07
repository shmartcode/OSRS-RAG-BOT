import os
import numpy as np
import faiss

# Paths
PROCESSED_DIR = "data/processed"
INDEX_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "vector_index.faiss")


def build_faiss_index():
    print("Loading embedding files...")

    # Find all embedding files saved previously
    embedding_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith("_embeddings.npy")])

    if not embedding_files:
        print("No embedding files found in data/processed!")
        return

    all_embeddings = []

    for file_name in embedding_files:
        file_path = os.path.join(PROCESSED_DIR, file_name)
        print(f"Loading {file_name}...")
        arr = np.load(file_path)
        all_embeddings.append(arr)

    # Combine all individual corpus embeddings into a single giant matrix
    matrix = np.vstack(all_embeddings).astype("float32")
    print(f"Total embedding matrix shape: {matrix.shape}")

    # Ensure vectors are normalized for cosine similarity via Inner Product
    print("Normalizing vectors...")
    faiss.normalize_L2(matrix)

    # Get dimensions
    num_vectors, dimension = matrix.shape
    print(f"Initializing FAISS IndexFlatIP for {num_vectors} vectors of dimension {dimension}...")

    # Create the Flat Inner Product index
    index = faiss.IndexFlatIP(dimension)

    print("Adding vectors to FAISS index...")
    index.add(matrix)

    # Save index to disk
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print(f"Successfully saved FAISS index to {INDEX_OUTPUT_PATH}!")


if __name__ == "__main__":
    build_faiss_index()
