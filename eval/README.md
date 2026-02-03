# MAGIC Evaluation

This directory is the evaluation entry point for the repository. It organizes safety evaluation, general capability evaluation, automated red-teaming, and attack pattern analysis in one place. To
keep results reproducible, create separate Conda environments for different tools.

We include the following tools:

- **Ai2 Safety Tool** for safety evaluation
https://github.com/allenai/safety-eval
- **OLMES (Open Language Model Evaluation System)** for general capability evaluation
https://github.com/allenai/olmes
- **OpenRT** for automated red-teaming evaluation
https://github.com/AI45Lab/OpenRT

## Directory layout

- `safety-eval-fork/`: safety evaluation and part of general capability evaluation
- `olmes/`: general capability evaluation
- `OpenRT/`: automated red-teaming
- `pattern/`: Attack Pattern extraction and analysis

## Environment setup

### Safety Eval

```bash
cd eval/benchmarks/safety-eval-fork
conda create -n safety-eval python=3.10
conda activate safety-eval
pip install -e .
pip install -r requirements.txt
pip install vllm==0.8.5
```
### OLMES
```bash
cd eval/benchmarks/olmes
conda create -n olmes python=3.10
conda activate olmes
pip install -e .
```
### OpenRT
```bash
cd OpenRT
conda create -n OpenRT python=3.10
conda activate OpenRT
pip install -e .
```
### Pattern

You can reuse the safety-eval environment.

## Evaluation notes

### Safety Eval

We evaluate the following tasks:

- Safety-related:
wildguardtest, harmbench_precompute, wildjailbreak:benign,
wildjailbreak:harmful, do_anything_now, harmbench,
or_bench:hard-1k, or_bench:toxic, xstest, strongreject
- General capability & instruction following:
alpacaeval, mmlu
- Multi-turn attacks:
x-teaming

You can run evaluations locally with GPU, e.g., harmbench.
To reproduce the MAGIC safety experiments:

- Use MODEL_TEMPLATE="game_defender" for MAGIC-Qwen models
- Use MODEL_TEMPLATE="hf" for MAGIC-llama models
- Deploy a Guard classifier model before running evaluations

Supported classifiers:

- CLASSIFIER="Qwen3GuardAPI" for Qwen3Guard
- CLASSIFIER="WlidGuardAPI" for WildGuard

Example:
```bash
CLASSIFIER="Qwen3GuardAPI"
export WILDGUARD_API_ENDPOINT="$API_URL"
export WILDGUARD_API_KEY="$API_KEY"
MODEL_TEMPLATE="game_defender"

python -u evaluation/eval.py generators \
--model_name_or_path "$MODEL_PATH" \
--model_input_template_path_or_name "$MODEL_TEMPLATE" \
--tasks "harmbench" \
--report_output_path "$safety_dir/metrics.json" \
--save_individual_results_path "$safety_dir/all_results.json" \
--use_vllm \
--classifier_model_name "$CLASSIFIER_MODEL_NAME" \
--batch_size 4 2>&1 | tee "$safety_dir/eval.log"
```
To evaluate all safety tasks at once, set:
```bash
TASKS="wildguardtest,harmbench_precompute,wildjailbreak:benign,wildjailbreak:harmful,do_anything_now,harmbench,or_bench:hard-1k,or_bench:toxic,xstest"
```
When testing mmlu, no Guard classifier is needed and MODEL_TEMPLATE="hf" should be used.

You can also deploy the defender model as an API and use use_defender_api:
```bash
python -u evaluation/eval.py generators \
--model_name_or_path "$DEFENDER_MODEL_NAME" \
--model_input_template_path_or_name "hf" \
--tasks "alpacaeval" \
--report_output_path "$alpaca_dir/metrics.json" \
--save_individual_results_path "$alpaca_dir/all_results.json" \
--use_defender_api \
--batch_size 1 2>&1 | tee "$alpaca_dir/eval.log"
```
x-teaming evaluation:

- Entry script: eval/safety-eval-fork/evaluation/tasks/generation/x-teaming/run_xteaming_benchmark.py
- Edit eval/safety-eval-fork/evaluation/tasks/generation/x-teaming/config/integration_config.yaml
to specify the API interface for strategy generation and attack execution
- Run:
```bash
python run_xteaming_benchmark.py
```
### OLMES

We evaluate the following tasks:

- General capability:
ifeval, gpqa, arc-c

Run in the olmes directory:

python -m oe_eval.run_eval \
--model-path "$MODEL_PATH" \
$TASK_ARGS \
--output-dir "$TASK_RESULTS_DIR" \
2>&1 | tee "$TASK_RESULTS_DIR/eval.log"

### OpenRT

We run automated red-teaming with:

- White-box methods: gcg
- Black-box methods: pair, tree, autodan, autodan_turbo_r

We also integrate two baseline defenses in the repo:
self-eval, smoothllm

A unified script is provided: eval/OpenRT/run_eval_all.sh.

Model requirements:

- autodan needs a target model and a judge model
- pair (PAIR), tree (TAP), autodan_turbo_r (AutoDAN-Turbo)
require an additional attacker model
- In MAGIC experiments, the attacker model is fixed to Qwen2.5-7B-IT
- For the white-box method gcg, set LOCAL_MODEL_PATH

For MAGIC models and base models as target models, only the target model API URL needs to be changed:
```bash
TARGET_MODEL="DEFENDER_API_MODEL"
```
Smoothllm and Self-eval are inference_defense methods.
To enable them, add the prefix before the defender model name, e.g.:
```bash
TARGET_MODEL="selfeval-${DEFENDER_API_MODEL}"
```
### Pattern

Implements the Attack Pattern analysis described in Appendix F, and supports classification and statistics for rewritten prompts that follow the required format.
See eval/pattern/README.md for details.

You must deploy a Classify model (via API) for semantic classification.
The model follows templates in eval/pattern/src/prompts_grained.py and classifies rewritten attacks into 20 single or hybrid strategies.

If you use run_classification.sh, ensure the input JSON format is:
```json
{
"question": "Vanilla harmful question here",
"step": 20,
"exchange_id": 0,
"attacker": {
    "parsed_think": "Adversarial think process produced by the attacker",
    "parsed_answer": "Adversarial prompt produced by the attacker"
}
}
```
Fill in parameters in eval/pattern/src/run_classification.sh to start Attack Pattern analysis.

## 🫶🏻 Acknowledge
    # OpenRT
    @article{wang2026openrt,
        title={OpenRT: An Open-Source Red Teaming Framework for Multimodal LLMs},
        author={Wang, Xin and Chen, Yunhao and Li, Juncheng and Wang, Yixu and Yao, Yang and Gu, Tianle and Li, Jie and Teng, Yan and Ma, Xingjun and Wang, Yingchun and others},
        journal={arXiv preprint arXiv:2601.01592},
        year={2026}
    }

    # Olmes
    @inproceedings{gu2025olmes,
        title={Olmes: A standard for language model evaluations},
        author={Gu, Yuling and Tafjord, Oyvind and Kuehl, Bailey and Haddad, Dany and Dodge, Jesse and Hajishirzi, Hannaneh},
        booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
        pages={5005--5033},
        year={2025}
    }

    # Ai2 safety tool
    @misc{wildteaming2024,
        title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models}, 
        author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar and Niloofar Mireshghallah and Ximing Lu and Maarten Sap and Yejin Choi and Nouha Dziri},
        year={2024},
        eprint={2406.18510},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2406.18510}, 
    }

    @misc{wildguard2024,
      title={WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs}, 
      author={Seungju Han and Kavel Rao and Allyson Ettinger and Liwei Jiang and Bill Yuchen Lin and Nathan Lambert and Yejin Choi and Nouha Dziri},
      year={2024},
      eprint={2406.18495},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18495}, 
    }
