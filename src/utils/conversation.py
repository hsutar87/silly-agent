class ConversationManager:
    def __init__(self):
        self.history = []

    def get_history(self) -> list:
        """Retrieve conversation history."""
        return self.history

    def add_message(self, role: str, message: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": message})

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
