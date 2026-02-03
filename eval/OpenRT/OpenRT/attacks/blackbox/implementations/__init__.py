"""
Blackbox attack implementations for the OpenRT.

This module contains implementations of various blackbox attacks against language models.
Each attack is automatically registered with the attack_registry when imported.
"""

from OpenRT.attacks.blackbox.implementations.autodan_turbo import AutoDANTurbo
from OpenRT.attacks.blackbox.implementations.autodan_turbo_r import AutoDANTurboR
from OpenRT.attacks.blackbox.implementations.genetic_attack import GeneticAttack
from OpenRT.attacks.blackbox.implementations.autodan import AutoDAN_Attack
from OpenRT.attacks.blackbox.implementations.pair_attack import PAIRAttack
from OpenRT.attacks.blackbox.implementations.renellm_attack import ReNeLLMAttack
from OpenRT.attacks.blackbox.implementations.gptfuzzer import GPTFuzzerAttack
from OpenRT.attacks.blackbox.implementations.cipherchat import CipherChatAttack
from OpenRT.attacks.blackbox.implementations.deepinception_attack import DeepInceptionAttack
from OpenRT.attacks.blackbox.implementations.ica_attack import ICAAttack
from OpenRT.attacks.blackbox.implementations.jailbroken_attack import JailBrokenAttack
from OpenRT.attacks.blackbox.implementations.xteaming_attack import XTeamingAttack
from OpenRT.attacks.blackbox.implementations.redqueen_attack import RedQueenAttack
from OpenRT.attacks.blackbox.implementations.actor_attack import ActorAttack
from OpenRT.attacks.blackbox.implementations.evosynth.evosynth_attack import EvosynthAttack
from OpenRT.attacks.blackbox.implementations.crescendo_attack import CrescendoAttack
from OpenRT.attacks.blackbox.implementations.direct_attack import DirectAttack
from OpenRT.attacks.blackbox.implementations.flipattack import FlipAttack
from OpenRT.attacks.blackbox.implementations.mousetrap import MousetrapAttack
from OpenRT.attacks.blackbox.implementations.multilingual_attack import MultilingualAttack
from OpenRT.attacks.blackbox.implementations.prefill_attack import PrefillAttack
from OpenRT.attacks.blackbox.implementations.rainbow_teaming import RainbowTeamingAttack
from OpenRT.attacks.blackbox.implementations.cipherchat import CipherChatAttack
from OpenRT.attacks.blackbox.implementations.coa import CoAAttack
from OpenRT.attacks.blackbox.implementations.magic_attack import MagicAttack
from OpenRT.attacks.blackbox.implementations.CodeAttack import CodeAttack
from OpenRT.attacks.blackbox.implementations.CSDJ import CSDJAttack
from OpenRT.attacks.blackbox.implementations.DrAttack import DrAttack
from OpenRT.attacks.blackbox.implementations.FigStep import FigStepAttack
from OpenRT.attacks.blackbox.implementations.gptfuzzer import GPTFuzzerAttack
from OpenRT.attacks.blackbox.implementations.HADES import HadesAttack
from OpenRT.attacks.blackbox.implementations.himrd import HimrdAttack
from OpenRT.attacks.blackbox.implementations.IDEATOR import IdeatorAttack
from OpenRT.attacks.blackbox.implementations.jood import JoodAttack
from OpenRT.attacks.blackbox.implementations.MML import MMLAttack
from OpenRT.attacks.blackbox.implementations.query_relevant import QueryRelevantAttack
from OpenRT.attacks.blackbox.implementations.race import RACEAttack


__all__ = [
    'AutoDANTurbo',
    'AutoDANTurboR',
    'GeneticAttack',
    'AutoDAN_Attack',
    'PAIRAttack',
    'ReNeLLMAttack',
    'GPTFuzzerAttack',
    'CipherChatAttack',
    'DeepInceptionAttack',
    'ICAAttack',
    'JailBrokenAttack',
    'XTeamingAttack',
    'RedQueenAttack',
    'ActorAttack',
    'EvosynthAttack',
    'CrescendoAttack',
    'DirectAttack',
    'FlipAttack',
    'MousetrapAttack',
    'MultilingualAttack',
    'PrefillAttack',
    'RainbowTeamingAttack',
    'CoAAttack',
    'MagicAttack',
    'CodeAttack',
    'CSDJAttack',
    'DrAttack',
    'FigStepAttack',
    'HadesAttack',
    'HimrdAttack',
    'IdeatorAttack',
    'JoodAttack',
    'MMLAttack',
    'QueryRelevantAttack',
    'RACEAttack',
]
