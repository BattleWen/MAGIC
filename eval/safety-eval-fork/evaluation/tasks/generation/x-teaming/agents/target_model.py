from typing import Dict

import tiktoken
from textgrad import Variable

from agents.base_agent import BaseAgent
from tgd import BlackboxLLMWithHistory, TGBaseAgentEngine


class TargetModel(BaseAgent):
    """Target model that responds to attacker's messages."""

    def __init__(self, config: Dict):
        """Initialize the target model with configuration."""
        super().__init__(config)
        self.messages = []

    def generate_response(self, attacker_message: str) -> str:
        """Generate a response to the attacker's message."""
        # Update message history
        if not self.messages:
            self.messages = [{"role": "user", "content": attacker_message}]
        else:
            self.messages.append({"role": "user", "content": attacker_message})

        # Generate response using exact temperature from config
        response = self.call_api(
            messages=self.messages, temperature=self.config["temperature"]
        )
        self.messages.append({"role": "assistant", "content": response})

        return response


class TGTargetModel(BaseAgent):
    """Variant of target model that responds to attacker's messages."""

    def __init__(self, config: Dict):
        """Initialize the target model with configuration."""
        super().__init__(config)
        self.target_model = BlackboxLLMWithHistory(TGBaseAgentEngine(config))
        self.messages = []

    def generate_response(self, attacker_message: Variable) -> Variable:
        """Generate a response to the attacker's message."""
        # Update message history
        if not self.messages:
            self.messages = [{"role": "user", "content": attacker_message.value}]
        else:
            self.messages.append({"role": "user", "content": attacker_message.value})

        response = self.target_model(attacker_message, history=self.messages)
        self.messages.append({"role": "assistant", "content": response.value})

        return response


def truncate_response(response_text: str, max_tokens: int = 512) -> str:
    """Truncates responses to prevent token overflow"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = encoding.encode(response_text)
        if len(tokens) <= max_tokens:
            return response_text
        return encoding.decode(tokens[:max_tokens])
    except Exception as e:
        print(f"Warning: Error in token counting: {e}")
        return response_text
