import logging
import os
from random import uniform
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

from colorama import Fore, Style
from openai import OpenAI


class APICallError(Exception):
    """Custom exception for API call failures"""

    pass


class BaseAgent:
    """Universal base agent for handling different LLM APIs.

    Configuration structure:
    {
        "provider": "openai|google|anthropic|sglang|ollama|together|openrouter",  # Required: API provider
        "model": "model-name",                                # Required: Model identifier
        "temperature": 0.7,                                   # Required: Temperature for sampling
        "max_retries": 3,                                     # Optional: Number of retries (default: 3)

        # Provider-specific configurations
        "project_id": "your-project",           # Required for Google
        "location": "us-central1",              # Required for Google
        "port": 30000,                          # Required for SGLang
        "base_url": "http://localhost:11434",   # Required for Ollama
        "api_key": "ollama",                    # Required for Ollama
        "response_format": {"type": "json_object"}  # Optional: For JSON responses (only for openai models) Make sure you include the word json in some form in the message
        "http_referer": "your-site-url",        # Optional for OpenRouter: Site URL for rankings
        "x_title": "your-site-name"             # Optional for OpenRouter: Site title for rankings
    }

    Usage:
        agent = BaseAgent(config)
        response = agent.call_api(messages, temperature=0.7)
        # Or with message inspection:
        response, messages = agent.call_api(messages, temperature=0.7, return_messages=True)
        # For JSON response (openai only):
        response = agent.call_api(messages, temperature=0.7, response_format={"type": "json_object"})
    """

    def __init__(self, config: Dict):
        """Initialize base agent with provider-specific setup."""
        self.config = config
        self.provider = config["provider"]
        self.max_retries = config.get("max_retries", 3)

        try:
            if self.provider == "openai":
                base_url = (
                    config.get("base_url")
                    or os.getenv("OPENAI_BASE_URL")
                    or "https://api.openai.com/v1"
                )
                api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY") or ""
                self.client = OpenAI(base_url=base_url, api_key=api_key)
                self.model = config["model"]
            elif self.provider == "sglang":
                base_url = (
                    config.get("base_url")
                    or f"http://localhost:{config.get('port', 30000)}/v1"
                )
                api_key = config.get("api_key") or "None"
                self.client = OpenAI(base_url=base_url, api_key=api_key)
                self.model = config["model"]
            else:
                raise APICallError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            raise APICallError(f"Error initializing {self.provider} client: {str(e)}")

    def call_api(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
        return_messages: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Universal API call handler with retries.

        Args:
            messages: List of message dictionaries
            temperature: Float value for temperature
            response_format: Optional response format specifications
            return_messages: If True, returns tuple of (response, messages)

        Returns:
            Either string response or tuple of (response, messages) if return_messages=True
        """
        print(
            f"{Fore.GREEN}Model is {self.model}, temperature is {temperature}{Style.RESET_ALL}"
        )

        # breakpoint()

        provider_configs = {
            "openai": {"base_delay": 1, "retry_delay": 3, "jitter": 1},
            "sglang": {"base_delay": 0, "retry_delay": 1, "jitter": 0.5},
        }

        config = provider_configs.get(self.provider, provider_configs["openai"])

        # print(json.dumps(messages, indent=2, ensure_ascii=False))

        for attempt in range(self.max_retries):
            try:
                # Add retry delay if needed
                if attempt > 0:
                    delay = config["retry_delay"] * attempt + uniform(
                        0, config["jitter"]
                    )
                    print(
                        f"\nRetry attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s..."
                    )
                    sleep(delay)

                if self.provider == "openai":
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if response_format:
                        api_params["response_format"] = response_format

                    response = self.client.chat.completions.create(**api_params)
                    response = response.choices[0].message.content
                elif self.provider == "sglang":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2048,
                    )
                    response = response.choices[0].message.content
                else:
                    raise APICallError(f"Unsupported provider: {self.provider}")

                # Return based on return_messages flag
                return (response, messages) if return_messages else response

            except Exception as e:
                error_msg = str(e)
                logging.error(
                    f"BaseAgent: API call failed for {self.provider} (Attempt {attempt + 1}/{self.max_retries}): {error_msg}",
                    exc_info=e,
                )
                if attempt == self.max_retries - 1:
                    raise APICallError(
                        f"BaseAgent: Failed to get response from {self.provider}: {error_msg}"
                    )
                continue

