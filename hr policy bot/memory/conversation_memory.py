# memory/conversation_memory.py

from typing import List, Dict


class ConversationMemory:

    def __init__(self, max_turns: int = 4):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})

    def get_recent_history(self) -> List[Dict[str, str]]:
        """
        Returns only the last N turns to control token growth.
        """
        return self.history[-self.max_turns:]

    def get_formatted_history(self) -> str:
        """
        Converts recent memory into readable text for prompt.
        """
        formatted = ""
        for msg in self.get_recent_history():
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return formatted

    def clear(self):
        self.history = []