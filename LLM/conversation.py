from config.logger_config import logger
import time

class ConversationChain:
    def __init__(self, remember_message_limit=10):
        self.messages = []
        self.remember_message_limit = remember_message_limit

    def add_message(self, message, sender="user"):
        """Add a message with timestamp and sender role."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.messages.append({
            "timestamp": timestamp,
            "sender": sender,
            "message": message
        })

        # Keep memory trimmed
        if len(self.messages) > self.remember_message_limit:
            self.messages = self.messages[-self.remember_message_limit:]

    def get_all_messages(self):
        """Return full conversation history."""
        return self.messages.reverse()

    def get_recent_messages(self):
        """Return only the recent N messages."""
        return self.messages[-self.remember_message_limit:]

    def get_formatted_context(self):
        """Return a plain text version of recent messages (for LLM input)."""
        return "\n".join(
            [f"{m['sender'].capitalize()}: {m['message']}" for m in self.get_recent_messages()]
        )

    def __str__(self):
        return self.get_formatted_context()