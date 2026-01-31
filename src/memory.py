from typing import List, Dict

class ConversationManager:
    """
    Manages the short-term memory of the Silly Agent.
    Uses a sliding window to keep the most recent context.
    """
    def __init__(self, max_history: int = 6):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Adds a message to the history and prunes if over limit."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Remove oldest message

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the current conversation thread."""
        return self.history

    def clear(self):
        """Wipes the memory bank."""
        self.history = []