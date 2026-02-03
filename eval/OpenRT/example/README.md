# OpenRT Examples

This folder contains examples of how to use different attack methods in the OpenRT Framework.

## Available Examples

- `actor_attack_example.py`: Demonstrates the ActorAttack method, a multi-agent coordination-based jailbreak attack leveraging role specialization and interaction.
- `autodan_attack_example_2024.py`: Implements the AutoDAN attack using hierarchical genetic algorithms for automated jailbreak prompt generation.
- `autodan_turbo_example.py`: Demonstrates AutoDANTurbo, an enhanced AutoDAN variant with turbo optimization for faster convergence.
- `autodan_turbo_r_example.py`: Implements AutoDANTurboR, combining hierarchical genetic algorithms with turbo optimization for robust jailbreak attacks.
- `cipherchat_attack_example_2024.py`: Demonstrates the CipherChat attack, which obfuscates malicious intent using cipher-based transformations.
- `coa_attack_example.py`: Implements the Chain-of-Action (CoA) attack that decomposes malicious intent into executable action sequences.
- `code_attack_example.py`: Demonstrates the CodeAttack method by transforming malicious instructions into code-style representations.
- `crescendo_attack_example.py`: Implements the Crescendo attack, which progressively escalates benign prompts into malicious ones across multiple turns.
- `csdj_attack_example.py`: Demonstrates the Composite Semantic Decomposition Jailbreak (CSDJ) method by decomposing and recombining semantic units.
- `deepinception_attack_example_2024.py`: Implements the DeepInception attack using multi-layered role-playing and nested contexts.
- `drattack_example.py`: Demonstrates DrAttack, an automated prompt engineering approach for discovering jailbreak strategies.
- `evosynth_example.py`: Implements EvoSynth, a code-level evolutionary synthesis attack based on mutation and selection.
- `figstep_attack_example.py`: Demonstrates the FigStep attack, which uses figure-based stepping stone prompts to bypass safety constraints.
- `flipattack_example.py`: Implements FlipAttack by flipping polarity or intent representations to evade safety detection.
- `gptfuzzer_attack_example_2023.py`: Demonstrates GPTFuzzer, a mutation-based fuzzing framework for large language model jailbreaks.
- `hades_attack_example.py`: Implements HADES, a visual vulnerability amplification attack targeting multimodal models.
- `himrd_attack_example.py`: Demonstrates HIMRD, a hierarchical multi-turn red-teaming attack with image generation.
- `ica_attack_example_2023.py`: Implements the In-Context Attack (ICA) by exploiting in-context learning behaviors.
- `ideator_attack_example.py`: Demonstrates the Ideator attack, which applies iterative design thinking with image generation.
- `jailbroken_attack_example_2023.py`: Implements JailBroken, a template-based jailbreak attack method.
- `jood_attack_example.py`: Demonstrates JOOD, a just-in-time adversarial prompt attack with image mixing.
- `mml_attack_example.py`: Implements MML, a cross-modal encryption-based jailbreak attack.
- `mousetrap_example.py`: Demonstrates the Mousetrap attack, a prompt injection technique exploiting hidden instructions.
- `multilingual_attack_example_2024.py`: Implements the Multilingual attack, which leverages cross-language prompt transfer.
- `nanogcg_attack_example.py`: Demonstrates NanoGCG, a lightweight gradient-based jailbreak attack.
- `pair_attack_example_2024.py`: Implements PAIR, an automatic iterative prompt refinement attack.
- `prefill_attack.py`: Demonstrates the Prefill attack by injecting malicious intent into pre-filled context.
- `query_relevant_attack_example.py`: Implements the Query-Relevant attack combined with diffusion-based image generation.
- `race_attack_example.py`: Demonstrates RACE, a multi-round adversarial refinement jailbreak method.
- `rainbow_teaming_attack_example.py`: Implements RainbowTeaming, a diverse multi-agent strategy-based attack.
- `redqueen_attack_example.py`: Demonstrates RedQueen, an adaptive prompt transformation jailbreak method.
- `renellm_attack_example.py`: Implements ReNeLLM, a neural-guided prompt optimization attack.
- `seqar_attack_example.py`: Demonstrates SeqAR, a sequential adversarial refinement attack.
- `si_attack_example.py`: Implements the SI attack based on shuffle inconsistency optimization.
- `tree_attack_example_2024.py`: Demonstrates TreeAttack, a tree-structured prompt evolution method.
- `visual_jailbreak_example.py`: Demonstrates a generic visual jailbreak attack targeting multimodal models.
- `xteaming_attack_2025.py`: Implements XTeaming, a multi-agent coordination-based red-teaming attack.

## Running Examples

To run an example:

```bash
python example/pair_attack_example_2024.py