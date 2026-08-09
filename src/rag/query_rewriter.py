import os
from openai import OpenAI

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.1:8b")


class QueryRewriter:

    def __init__(self, base_url: str = LOCAL_LLM_URL, model_name: str = LOCAL_MODEL_NAME):
        """rewrites follow up queries to allow re-querying the OSRS rag system without needing to specify the information from the original query"""
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="ollama")

        self.system_prompt = (
            "You are a search query reformulation assistant for an Old School RuneScape (OSRS) search engine.\n"
            "Your job is to analyze the recent conversation history and the latest user message.\n"
            "TOPIC SWITCHES: If the user question is a new, standalone question (e.g. asking about a completely different item, monster, or quest like 'what drops rune scimitars'), DO NOT inject previous entities (like 'abyssal whip') from chat history. Return the new question AS IS.\n"
            "PRONOUN RESOLUTION: Only rewrite if the question is a follow-up that contains pronouns or implicit context (e.g., 'how do I get it?', 'what are its stats?', 'where is that located?').\n"
            "If the user message is a follow-up rewrite it into a single, standalone search query that contains all necessary entity names (boss, monster, item, skill).\n"
            "If the user message is already self-contained, return it as-is.\n"
            "OUTPUT ONLY THE REWRITTEN QUERY. DO NOT ADD INTROS, OUTROS, OR EXPLANATIONS."
        )

    def rewrite_query(self, user_query: str, chat_history: list[dict]) -> str:
        """Reformulates short multi-turn user queries into standalone search queries."""
        # If there's no chat history, the prompt is standalone
        if not chat_history:
            return user_query

        # Construct payload with history context
        messages = [{"role": "system", "content": self.system_prompt}]

        # Include last few history turns for context
        for turn in chat_history[-4:]:
            messages.append(turn)

        messages.append(
            {
                "role": "user",
                "content": f"Formulate a standalone search query for this follow-up: '{user_query}'",
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,  # Zero temperature for deterministic rewrites
            )
            rewritten = response.choices[0].message.content.strip()
            # Fallback check if LLM returned an empty string
            return rewritten if rewritten else user_query
        except Exception:
            # If local LLM fails, safely fallback to raw query
            return user_query
