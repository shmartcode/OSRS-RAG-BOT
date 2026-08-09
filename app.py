import os
import sys
import warnings

# Suppress non-critical library warnings during startup
warnings.filterwarnings("ignore")

from typing import Any, Dict
from openai import OpenAI
from src.rag.context_formatter import build_rag_prompt
from src.rag.query_rewriter import QueryRewriter
from src.rag.retriever import LocalRAGRetriever
from src.rag.session_manager import SessionManager

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://ollama:11434/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.1:8b")


class OSRSApp:

    def __init__(
        self,
        base_url: str = LOCAL_LLM_URL,
        model_name: str = LOCAL_MODEL_NAME,
        top_k_hits: int = 4,
    ):
        self.model_name = model_name
        self.top_k_hits = top_k_hits

        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.retriever = LocalRAGRetriever()
        self.session_manager = SessionManager()
        self.query_rewriter = QueryRewriter(base_url, model_name)

    def process_query(self, user_query: str, session_id: str = "default_user") -> str:
        session = self.session_manager.get_or_create_session(session_id)

        # 1. Fetch existing chat history BEFORE adding the new message
        chat_history = session.get_chat_history()

        # 2. Rewrite query (only if chat_history exists)
        if chat_history:
            search_query = self.query_rewriter.rewrite_query(user_query, chat_history)
        else:
            search_query = user_query

        # Debug log to verify what query is actually being searched
        print(f"\n[DEBUG] Original: '{user_query}' -> Rewritten: '{search_query}'")

        # 3. Add original user message to session
        session.add_user_message(user_query)

        # 4. Search vector DB with rewritten query
        hits = self.retriever.search(search_query, self.top_k_hits)
        session.set_last_hits(hits)

        # 5. Build prompt with retrieved hits
        full_prompt = build_rag_prompt(user_query=user_query, hits=hits)

        # 6. Build messages payload for LLM
        messages = [{"role": "system", "content": "You are an expert OSRS assistant. Answer strictly based on the context."}]

        # Append previous turns
        for turn in chat_history:
            messages.append(turn)

        # Append current user prompt with context
        messages.append({"role": "user", "content": full_prompt})

        # 7. Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0.2, extra_body={"num_ctx": 2048}  # Keep VRAM usage low for fast inference
            )
            assistant_response = response.choices[0].message.content.strip()
        except Exception as e:
            assistant_response = f"[Error contacting local LLM: {e}]"

        session.add_assistant_message(assistant_response)
        return assistant_response


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("Initializing OSRS RAG Assistant...")

    # Initialize the app once at startup
    try:
        app = OSRSApp()
        print("✅ Ready! Ask a question or type 'exit' to quit.\n")
    except Exception as e:
        print(f"❌ Failed to initialize app: {e}")
        sys.exit(1)

    session_id = "default_user"

    while True:
        try:
            user_query = input("OSRS-Bot > ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting RAG Bot. Goodbye!")
                break

            print("\nThinking & Retrieving...")
            response = app.process_query(user_query, session_id)

            print(f"\nAssistant:\n{response}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting RAG Bot. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")


if __name__ == "__main__":
    main()
