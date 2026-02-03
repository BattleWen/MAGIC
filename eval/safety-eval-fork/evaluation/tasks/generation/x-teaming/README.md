# 𝕏-Teaming: Multi-Turn Jailbreaks and Defenses with Adaptive Multi-Agents
by *Salman Rahman\*, Liwei Jiang\*, James Shiffer\*, Genglin Liu, Sheriff Issaka, Md Rizwan Parvez, Hamid Palangi, Kai-Wei Chang, Yejin Choi, Saadia Gabriel*

<small>*\*equal contribution*</small>

<div align="center">
    <a href="https://arxiv.org/abs/2504.13203"> <img alt="paper link" src="https://img.shields.io/badge/arXiv-2504.13203-b31b1b.svg"></a>
    <a href="https://huggingface.co/datasets/marslabucla/XGuard-Train"> <img alt="dataset link" src="https://img.shields.io/badge/HuggingFace-XGuard--Train-yellow.svg"></a>
    <a href="https://x-teaming.github.io"> <img alt="dataset link" src="https://img.shields.io/badge/Website-X--Teaming-blue.svg"></a>
    <br>
    <br>
    <img height="400" src="figure.png">
</div>

<br>

**𝕏-Teaming** is a scalable framework that systematically explores how seemingly harmless interactions with LLMs escalate into harmful outcomes. It employs collaborative agents for planning, attack optimization, and verification, achieving state-of-the-art multi-turn jailbreak effectiveness and diversity with **success rates up to 98.1%** across representative leading open-weight and closed-source models. In particular, **𝕏-Teaming** achieves a **96.2% attack success rate** against the latest Claude 3.7 Sonnet model, which has been considered nearly immune to single-turn attacks.

## Quickstart

To test our jailbreaking method with your own prompts, use the provided `demo.ipynb` notebook. You will need at least an OpenAI API key to run the Verifier.

## Setup

1. Install requirements:
```bash
conda env create -f environment.yml
conda activate x-teaming
```

2. Configure `config/config.yaml` with an attacker, target, and TextGrad model. You can choose amongst several APIs or running locally with SGLang or Ollama. For example:

```yaml
attacker:
  provider: "sglang"
  model: "Qwen/Qwen2.5-32B-Instruct"
  port: 30000
  # ...

target:
  provider: "openai"
  model: "gpt-4o"
  # ...

textgrad:
  enabled: true
  provider: "openrouter"
  model: "deepseek/deepseek-chat"
  # ...
```

Refer to `agents/base_agent.py` for a complete list of supported APIs and model settings.

3. Set your OpenAI API key, as well as any additional API keys used, with environment variables:
```bash
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```

\* An OpenAI key is always required for the GPT-4o Verifier.

## Usage

### 1. Generate Attack Plans

The list of harmful behaviors is expected as a CSV file which is structured like the HarmBench dataset. At a minimum, it needs to have `Behavior`, `BehaviorID`, and `FunctionalCategory` columns.

```bash
# the default config assumes your csv is under "behaviors/"
mkdir behaviors
cd behaviors
# download HarmBench test set
wget https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_test.csv
```

By default, the script will only use the behaviors with a FunctionalCategory of `standard`.

Configure the generation settings:

```yaml
# config/config.yaml
attack_plan_generator:
  provider: "openai"
  model: "gpt-4o"
  # ...
  behavior_path: "./behaviors/harmbench_behaviors_text_test.csv"  # path to your CSV
  num_behaviors: 159   # this is the number of standard behaviors in the CSV, but can be set lower
```

Generate attack strategies:

```bash
python generate_attack_plans.py
```

This will:
- Use `num_behaviors` standard behaviors randomly selected from the aforementioned CSV
- Use prompts from `config/prompts/plan_generation_prompts.yaml`
- Generate diverse attack strategies based on config in `config/config.yaml`
- Save plans to `strategies/[timestamp]/attack_plans.json`

Next, we need to make sure the attacker can find your strategies. You can either update the following line in `config/config.yaml`:

```yaml
attacker:
  plans_file: './strategies/2025-04-01_12-34-56/attack_plans.json'
```

or copy the file to the preset path:

```bash
cp strategies/2025-04-01_12-34-56/attack_plans.json strategies/attack_plans.json
```

<details>
<summary><i>Extra: Analyze Diversity of Attack Strategies</i></summary>

Run the following script to generate heatmaps of pairwise embedding dissimilarities among the 10 most diverse attack plans per behavior, saved under the `diversity_stats/` subdirectory. It also produces a `mean_diversity.json` file containing the average strategy diversity for each behavior.

```bash
python analytics/strategy_diversity.py strategies/[timestamp]/attack_plans.json
```

<div align="center">
  <img alt="Heatmap of strategy diversity" src="strategy-diversity.png" height="400">
</div>
</details>

### 2. Execute Attacks

Run the attacks:

```bash
python main.py
```

Optionally, add the `--config <CONFIG_PATH>` flag to use an alternative config file.

This executes:
1. Attacker (`agents/attacker_agent.py`):
   - Uses prompts from `config/prompts/attacker_agent_prompts.yaml`
   - Executes adaptive multi-turn attack strategies
   - **Prompt Optimizer:** Uses TextGrad to optimize its queries based on verifier feedback (see `TGAttackerAgent` class)
   - **Planner:** Can extend attack plans with additional phases on-the-fly

2. Target Model (`agents/target_model.py`):
   - Responds to attacker messages
   - Uses configuration from `config/config.yaml`

3. Verifier (`agents/gpt_evaluator.py`):
    - Scores responses on a scale from 1-5 (benign to full jailbreak) and provides a reason
    - Uses GPT-4o
    - Truncates target responses under verification to 512 tokens by default

4. Results:
   - Saves conversation logs and results in `attacks/[timestamp]/`
   - Includes evaluation scores for each turn
   - Includes a copy of the configuration

### 3. View Results

Wait for the attack to finish, then run this script, which will display a summary of the attack including the attack success rate (ASR), the configuration it was run with, and the average number of turns taken and plans used per successful jailbreak.

```bash
python analytics/metrics.py <TIMESTAMP>
```

More detailed statistics, such as a category-wise ASR breakdown, and an exhaustive list of the number of turns taken for every successful jailbreak, can be viewed by adding the `-v` flag.


<details>
<summary><i>Extra: Analyze Diversity of Multi-Turn Attacks</i></summary>

Given the path to a results file, the following script will generate a `diversity_stats/mean_diversity.json` file containing the average multi-turn attack query diversity for each behavior.

**Note:** this only considers behaviors which had multiple successful jailbreak strategies (the exact number depending on the `-k` flag). This requires running an attack with the setting `run_all_strategies: true`.

```bash
python analytics/xteaming_diversity.py attacks/[timestamp]/all_results.json
```
</details>

## Configuration

Key files:
- `config/config.yaml`: Model and execution settings
- `config/prompts/`: Contains all prompt templates
  - `attacker_agent_prompts.yaml`: Multi-turn attack prompts
  - `plan_generation_prompts.yaml`: Strategy generation prompts

### Debug Mode

Run with limited logging:
```bash
python main.py  # debug=False by default
```

For complete logging, which includes the contents of every API call:
```bash
python main.py --debug
```

### Aggressive Mode

One can increase the attack parameters to improve the attack success rate. In particular, we changed these values:

```bash
# config/config.yaml
attacker:
  # ...
  max_turns: 10
  # ...
  strategies_per_behavior: 50

textgrad:
  enabled: true
  # ...
  max_turns_per_phase: 7
```

This achieved a 96.2% ASR against Claude 3.7 Sonnet. Among the other models we tested with, these settings almost always resulted in a 100% ASR.

### Quick Mode

The recommended behavior is to run only as many strategies as necessary to calculate the ASR.
```bash
# config/config.yaml
attacker:
   # ...
   run_all_strategies: false     # Only require one successful strategy per behavior
   strategies_per_behavior: 10   # Randomly sampled from the attack plans file
   # ...
```

For a complete evaluation of all strategies, use these options:
```bash
# config/config.yaml
attacker:
   # ...
   run_all_strategies: true
   strategies_per_behavior: 50   # 5 sets with 10 plans each;
                                 # refer to generate_attack_plans.py and
                                 # config/prompts/plan_generation_prompts.py
   # ...
```

### Multithreading

```yaml
# config/config.yaml
multithreading:
  max_workers: 20
```

Most of our scripts process different behaviors in parallel, which speeds up the overall experiments. Feel free to reduce this setting if you are hitting API rate limits.


# 𝕏Guard-Train: A Large-Scale Dataset for Multi-Turn LM Safety

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/marslabucla/XGuard-Train)

A comprehensive multi-turn safety dataset for improving conversational AI defenses against sophisticated jailbreaking attacks.

## Setup

Configuration file is located at `xguard/config.yaml`.

## Instructions for Data Generation

### 1. Generate Attack Plans

We sampled 10,000 harmful behaviors from WildJailbreak’s vanilla harmful collection and put them in a CSV (not included). The CSV is expected to have the following columns: `Behavior`, `id`.

Point the config to the path for the CSV:

```yaml
# xguard/config.yaml
attack_plan_generator:
  # ...
  behavior_path: "../behaviors/wildjailbreak.csv"
```

```bash
cd xguard
python generate_attack_plans.py
```

This uses a special config file `xguard/config.yaml` which generates five strategies per behavior in only one set.

The attack plans will end up in `xguard/output/[timestamp]/attack_plans.json`, instead of under the regular `attacks` folder.

### 2. Execute Attacks

The provided 𝕏Guard-Train config will run a fixed number of attack plans instead of skipping to the next behavior after the first successful one. Consequently, this will take significantly longer than the usual attacking process.

First, point the 𝕏Guard-Train config file to your latest `attack_plans.json`.

```yaml
# xguard/config.yaml
attacker:
  # ...
  plans_file: './xguard/output/[timestamp]/attack_plans.json'
  # ...
```

Then run `main.py`:

```bash
cd ..
python main.py -c xguard/config.yaml
```

By default, only two of the five strategies per behavior are actually ran. If you wish to change that, refer to these settings:

```yaml
# xguard/config.yaml
attacker:
  # ...
  strategies_per_behavior: 2     # randomly sampled out of 5
  # new fields for XGuard-Train
  sets_per_behavior: 1           # do not change
  strategies_per_set: 2          # change this if changing strategies_per_behavior
```

Make note of the path in which the run is saved, i.e. `attacks/[timestamp]/all_results.json`.

### 3. Generate clean responses

For those attack plans which did not already score a 1/5, we need to modify the conversation to end with a relevant refusal message for the purposes of 𝕏Guard-Train.

```bash
cd xguard
python generate_safe_responses.py [timestamp]
```

You can use the `-t` flag to increase the score cutoff from 1/5.

GPT-4o is used by default to generate these refusals. Change these settings if you wish to use a different model:

```yaml
# xguard/config.yaml
safety_data_generator:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0
```

The resulting conversations will be found in `xguard/output/[timestamp]/safe_convos_scores.json` (different timestamp from the source data).

### 4. Reformat JSON for SFT

This script can be run on an attack conversations file (`all_results.json`) or clean conversations file (`safe_convos_scores.json`) to reformat it into ShareGPT format for supervised finetuning, like so:

```json
[
   {
      "conversations": [
         {"from": "human", "value": "Hello world"},
         {"from": "gpt", "value": "Foobar"},
         ...
      ]
   },
   ...
]
```

Conversations which are found to have a final message with a score greater than 1 are not included in the output. (Likewise, this cutoff can be changed with `-t`.) Therefore, we compile our training data like so:

```bash
# export the already safe conversations from all_results into ShareGPT format
python reformat_convos.py ../attacks[timestamp1]/all_results.json ./output/train-1.json
# export the rewritten safe conversations from safe_convos_scores into ShareGPT format
python reformat_convos.py ./output/[timestamp2]/safe_convos_scores.json ./output/train-2.json
# merge the two (requires `jq`)
jq -s 'add' train-1.json train-2.json > train.json
```

## Ethics Statement

We acknowledge the dual-use nature of our work on 𝕏-Teaming and 𝕏Guard-Train, which
demonstrates significant vulnerabilities in current language models through multi-turn
attack methodologies. While these findings could potentially be misused, we believe
open-sourcing our research is essential to advance AI safety. We believe the benefits of
accelerating advances in multi-turn safety alignment significantly outweigh the marginal
risks of public release, particularly as these vulnerabilities would likely be discovered
independently by motivated actors. Our work represents a substantial effort to ensure
safety research keeps pace with rapidly evolving language model capabilities, ultimately
contributing to the development of more robust and trustworthy AI systems.
