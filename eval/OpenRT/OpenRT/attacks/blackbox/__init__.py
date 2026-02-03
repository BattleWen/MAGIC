"""
Blackbox attack implementations for the OpenRT.

This module contains implementations of various blackbox attacks against language models.
Each attack is automatically registered with the attack_registry when imported.
"""

# Import all blackbox attack implementations here
from .base import BaseBlackBoxAttack

from .implementations import *

# You can also use __all__ to control what's imported with "from implementations import *"
__all__ = [
    "BaseBlackBoxAttack",
    *implementations.__all__,
]