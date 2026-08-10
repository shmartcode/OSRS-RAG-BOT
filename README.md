# OSRS RAG Bot Intro Summary
An intelligent, offline Retrieval-Augmented Generation (RAG) assistant for Old School RuneScape (OSRS). Built with custom PyTorch Lightning embedding models, FAISS vector search, and a local Llama 3.1 8B LLM running on Ollama inside Docker.

# Intro details
This project was a learning experience in Machine Learning. My goals were to learn how to train my own model, how to utilize docker, how to pull and parse large quantity of data using an API, and how to make a model that can run offline once intialized.

## Quickstart
### 1. Clone the Repository
```bash
git clone [https://github.com/shmartcode/OSRS-RAG-BOT.git](https://github.com/shmartcode/OSRS-RAG-BOT.git)
```
### 2. Navigate to direcory and run "run.py"
* Navigate to the location you have cloned/saved the project and execute "run.py"
* run.py will build the docker container downloading all necessary packages including Llama 3.1
* first time executing "run.py" can take 10+ minutes depending on download speeds to download and install necessary packages as well as download and parse the required data.
* due to the intially long run times ive left progress bard and other would be debug information active to show progression
* the project is set to run on your gpu if available for faster computations


## Key Features
* **Custom Fine-Tuned Embeddings:** Retrained PyTorch Lightning embedder fine-tuned on OSRS Wiki dumps and osrsreboxed.
* **Smart Query Rewriting:** Converts multi-turn follow-up questions (e.g., *"What about budget options?"*) into standalone search queries based on session history.
* **Multi-Source Context:** Formats OSRS Wiki articles, drop tables, item stats, and monster stats into structured prompt context.
* **100% Local & Offline Stack:** Runs locally using Docker Desktop—no external API keys or cloud dependencies required.


## Prerequisites

Before running the project, make sure you have installed:

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (Ensure Docker Desktop is running before launching).
2. **Python 3.10+** (Used only to run the universal `run.py` launcher script).


## Notes on built docker container size
* Docker Image ~10.7gb
    * ~6.5gb : Pytorch and Nvidia CUDA packages 
    * ~3.2gb : Base OS and other packages 
    * ~1.0gb : Code, embeddings, index
* ollama 3.1 model: ~6.2gb.
* Total Size of docker container ~17gb


## Freeing up your disk on windows
* I've used the project on a Windows 11 machine. This means docker is running through WSL and uses virtual storage.
* Because of this docker's storage foot print grows as you run containers that install/take up data. You need to manually free that disk space if you no longer want to use the project.
* You can free your storage space on Windows 11 by
    - Shutting down docker
    - On windows Pro in terminal/powershell run "Optimize-VHD -Path "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx" -Mode Full"
    - On windows Home in terminal/powershell run "diskpart" then run:
            - "select vdisk file="C:\Users\YOUR_USERNAME\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
                    * your file path and folder names my differ please check.
            - "attach vdisk readonly"
            - "compact vdisk"
            - "detach vdisk"
            - "exit"

## Future possibilities/Issue Fixes
* Pulling live wiki prices for item (currently uses osrs wiki prices pulled in August 2026)
* pulling all data directly from osrs wiki instead of osrsreboxed which has some outdated info (and missing info as it has not been updated enough.)
* better quest guide information. currently the modelmay skip steps (conciseness side effect)
* set up parent/child chunk to help with follow up prompting. (asking what monsters drop an item and then asking for more monsters that drop that item
* rarity sorting does not account for different types of drop tables.)


## Note on Multi-source context
* I tried to use a combination of raw wiki text data and a supposedly popular package for OSRS information called osrsreboxed-db in order to create a faster model. The database would have allowed for reading and analyzing smaller bits of data compared to entire wiki page texts. This database also organized data making it easily accesible. The last commit for this project was January 2025. I later discovered that this "last commit" did not mean the data they used was updated as of then. With this I found that monsters that have been in the game since 2018 are not present in the osrsreboxed-db. Because of this as experience OSRS users test this project they will discover missing data or incorrect answers due to the missing data.

## Note on project testing and testing files
* Stage 1: I tested the project along the way with manual querying into the trained model. This part let me see the top results that the queries produced. Ideally the correct answer was result #1, or at least within top_k.
* Stage 2: At full build the system includes ollama 3.1 LLM to help generate user friendly responses. the LLM takes in the top_k query results and searches through them for the correct answer
* Some results in stage 1 produced false negative Fails due to parsing issues I have not yet resolved. This is seen in queries that require wiki text for answer. wiki text parsing is a pain in the butt. For example in stage 1 "query": "What are the rune costs for Ice Barrage?", porduced a FAIL but on full launch and test within the RAG system the query produces correct response.
