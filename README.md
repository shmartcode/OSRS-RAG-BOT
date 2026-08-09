# OSRS RAG Bot

An intelligent, offline Retrieval-Augmented Generation (RAG) assistant for Old School RuneScape (OSRS). Built with custom PyTorch Lightning embedding models, FAISS vector search, and a local Llama 3.1 8B LLM running on Ollama inside Docker.

## Quickstart
### 1. Clone the Repository
```bash
git clone [https://github.com/shmartcode/OSRS-RAG-BOT.git](https://github.com/shmartcode/OSRS-RAG-BOT.git)
```
### 2. Navigate to direcory and run "run.py"
cd OSRS-RAG-BOT
python run.py


## Key Features

* **Custom Fine-Tuned Embeddings:** Retrained PyTorch Lightning embedder fine-tuned on OSRS Wiki dumps and osrsreboxed.
* **Smart Query Rewriting:** Converts multi-turn follow-up questions (e.g., *"What about budget options?"*) into standalone search queries based on session history.
* **Multi-Source Context:** Formats OSRS Wiki articles, drop tables, item stats, and monster stats into structured prompt context.
* **100% Local & Offline Stack:** Runs locally using Docker Desktop—no external API keys or cloud dependencies required.


## Prerequisites

Before running the project, make sure you have installed:

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Ensure Docker Desktop is running before launching).
2. **Python 3.10+** (Used only to run the universal `run.py` launcher script).


## Future possibilities
* Pulling live wiki prices for item (currently uses osrs wiki prices pulled in August 2026)
* 
