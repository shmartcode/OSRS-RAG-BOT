import time
from typing import List, Dict, Optional


class ConversationSession:

    def __init__(self, session_id: str, max_turns: int = 10, ttl_seconds: int = 3600):
        self.session_id = session_id
        self.max_turns = max_turns  # Max chat turns to keep in context window
        self.ttl_seconds = ttl_seconds
        self.last_active = time.time()
        self.history: List[Dict[str, str]] = []
        self.last_retrieved_hits: List[Dict] = []

    def add_user_message(self, message: str):
        self._check_expiration()
        self.history.append({"role": "user", "content": message})
        self._trim_history()

    def add_assistant_message(self, message: str):
        self._check_expiration()
        self.history.append({"role": "assistant", "content": message})
        self._trim_history()

    def set_last_hits(self, hits: List[Dict]):
        """Saves the last retrieved vector hits for context pinning or inspectability."""
        self.last_retrieved_hits = hits

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Returns full chat history for LLM message payloads."""
        return self.history

    def _trim_history(self):
        """Keep history within sliding window bounds (2 entries per turn: user + assistant)."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2) :]

    def _check_expiration(self):
        """Resets session if idle longer than TTL."""
        if time.time() - self.last_active > self.ttl_seconds:
            self.history = []
            self.last_retrieved_hits = []
        self.last_active = time.time()


class SessionManager:

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str = "default_user") -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
