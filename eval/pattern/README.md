# Attack Pattern Extraction (RL)

This README covers:
1) data preparation (required JSONL format + example),
2) how to run `run_rl_classification.sh`,
3) how to run `src/analyze_pattern_diversity.py`,
4) a brief note on the extraction prompt file.

## 1. Data preparation

Prepare a JSONL file for each RL training step, for example:
`extracted_step_20.jsonl`, `extracted_step_40.jsonl`, ...

Each line is a JSON object with at least the following fields:

```json
{
  "question": "Vanilla harmful question here",
  "step": 20,
  "exchange_id": 0,
  "attacker": {
    "parsed_think": "Adversarial think prossess produced by the attacker",
    "parsed_answer": "Adversarial prompt produced by the attacker"
  }
}
```

Required fields for extraction:
- `question`: original benign question.
- `step`: RL training step for this exchange.
- `exchange_id`: unique id within the step (any integer).
- `attacker.parsed_answer`: the attacker prompt to be classified.

You can keep additional fields; they will be preserved in the output, eg. eval/pattern/raw_json/extracted_step_20.jsonl

## 2. Run RL fine-grained classification

Use `run_classification.sh` to classify attacker prompts with the fine-grained schema.

```bash
bash run_classification.sh \
  --input-dir /path/to/extracted_jsonl_dir \
  --output-dir /path/to/rl_grained
```

Notes:
- `--input-dir` should contain `extracted_step_*.jsonl`.
- The script writes classified JSONL files to `--output-dir`.
- Don't forget use your classification model

Some small results stored in eval/pattern/results/rl_grained/classified_extracted_step_20.jsonl

## 3. Visualize pattern diversity

Use the visualization script after classification:

```bash
python src/analyze_pattern_diversity.py \
  --sft-result /path/to/sft_grain_result.json \
  --rl-classified-dir /path/to/rl_grained \
  --output-dir /path/to/figures
```

This script compares SFT vs. RL fine-grained distributions and produces plots for diversity and mixture trends.

## 4. Extraction prompt file

`src/prompts_grained.p` defines the fine-grained attack taxonomy and the classification prompt template. The taxonomy contains 20 fine-grained classes (IDs 0–19).
