# memory/conversation_memory.py

from typing import List, Dict


class ConversationMemory:
    """
    Stores conversation history within session.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def get_formatted_history(self) -> str:
        """
        Converts memory into readable text for prompt.
        """
        formatted = ""
        for msg in self.history:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return formatted

    def clear(self):
        self.history = []