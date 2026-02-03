"""
CipherChat attack implementation for the OpenRT.

This attack uses ciphers to obfuscate harmful prompts, allowing them to bypass
safety filters while still being understood by the LLM through in-context learning.
"""

from .attack import CipherChatAttack

__all__ = ["CipherChatAttack"]