from sentence_transformers import SentenceTransformer

# 1. Load your local model folder
model = SentenceTransformer("./fine_tuned_osrs_embedder_v2")

# 2. Push to your Hugging Face account
# Format: "your-hf-username/repository-name"
model.push_to_hub("shmartcode/osrs-embedder-v2", private=False)
