import json
import os
import random
import threading
import time
from typing import List, Union

import platformdirs
import textgrad as tg
from tenacity import retry, stop_after_attempt, wait_fixed
from textgrad import Variable
from textgrad.engine import CachedEngine, EngineLM
from textgrad.engine.openai import ChatOpenAI

from agents.base_agent import BaseAgent

_TEXTGRAD_CACHE_LOCK = threading.Lock()


class ChatOpenAIWithHistory(ChatOpenAI):
    """
    Adapted from github.com/zou-group/textgrad/issues/116
    """

    def __init__(self, *args, **kwargs):
        self.history_messages = []
        super().__init__(*args, **kwargs)

    def inject_history(self, messages: list[dict]) -> None:
        self.history_messages = messages

    def _generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            *self.history_messages,
            {"role": "user", "content": prompt},
        ]
        self.history_messages.clear()
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def _generate_from_multiple_input(
        self,
        content: List[Union[str, bytes]],
        system_prompt=None,
        temperature=0,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            *self.history_messages,
            {"role": "user", "content": formatted_content},
        ]
        self.history_messages.clear()
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text


class BlackboxLLMWithHistory(tg.BlackboxLLM):
    """
    Adapted from github.com/zou-group/textgrad/issues/116
    """

    def forward(self, x: Variable, history: list[dict] = []) -> Variable:
        if history and hasattr(self.engine, "inject_history"):
            self.engine.inject_history(history)

        return self.llm_call(x)


class TGBaseAgentEngine(EngineLM, CachedEngine):
    """A TextGrad-compatible LLM engine, like ChatOpenAIWithHistory, but using our BaseAgent under the hood."""

    def __init__(self, config: dict):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(
            root,
            f"cache_{config['provider']}_{config['model'].replace('/', '_')}_{os.getpid()}.db",
        )
        super().__init__(cache_path=cache_path)

        self._base_agent = BaseAgent(config)
        self.temperature = config["temperature"] if "temperature" in config else 0
        self.history_messages = []

    def inject_history(self, messages: list[dict]) -> None:
        self.history_messages = messages

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def generate(self, prompt, system_prompt=None, **kwargs):
        sys_prompt_arg = (
            system_prompt
            if system_prompt
            else "You are a helpful, creative, and smart assistant."
        )
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            *self.history_messages,
            {"role": "user", "content": prompt},
        ]
        self.history_messages.clear()
        response = self._base_agent.call_api(messages, self.temperature)
        cache_key = sys_prompt_arg + prompt
        for attempt in range(10):
            try:
                with _TEXTGRAD_CACHE_LOCK:
                    self._save_cache(cache_key, response)
                break
            except Exception as e:
                if "database is locked" not in str(e).lower():
                    raise
                time.sleep(min(1.0, 0.05 * (attempt + 1)) + random.random() * 0.05)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
