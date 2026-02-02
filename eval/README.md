# MAGIC Evaluation

This directory is the evaluation entry point for the repository. It organizes safety evaluation, general capability evaluation, automated red-teaming, and Attack Pattern analysis in one place. To keep results reproducible, create separate Conda environments for different tools. 🙂‍↕️

## Directory layout

- `benchmarks/safety-eval-fork/`: safety evaluation and part of general capability evaluation
- `benchmarks/olmes/`: general capability evaluation
- `benchmarks/OpenRT/`: automated red-teaming
- `pattern/`: Attack Pattern extraction and analysis

## Environment setup

### Safety Eval

```bash
cd eval/benchmarks/safety-eval-fork
conda create -n safety-eval python=3.10
conda activate safety-eval
pip install -e .
pip install -r requirements.txt
pip install vllm==0.4.2
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
git clone https://github.com/AI45Lab/OpenRT.git
cd OpenRT
pip install -e .
```

### Pattern

You can reuse the `safety-eval` environment.

## Evaluation notes

### Safety Eval

We evaluate the following tasks:

- Safety-related:
  `wildguardtest`, `harmbench_precompute`, `wildjailbreak:benign`,
  `wildjailbreak:harmful`, `do_anything_now`, `harmbench`,
  `or_bench:hard-1k`, `or_bench:toxic`, `xstest`, `strongreject`
- General capability and instruction following:
  `alpacaeval`, `mmlu`
- Multi-turn attacks:
  `x-teaming`

### OLMES

We evaluate the following tasks:

- General capability:
  `ifeval`, `gpqa`, `arc-c`

### OpenRT

We run automated red-teaming with:

- White-box methods:
  `gcg`
- Black-box methods:
  `pair`, `tree`, `autodan`, `autodan_turbo_r`

We also integrate two baseline defense methods in the repo:
`self-eval`, `smoothllm`

### Pattern

Implements the Attack Pattern analysis described in Appendix F, and supports
classification and statistics for rewritten prompts that follow the required
format.
