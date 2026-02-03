import abc
import json
import os
import re
from abc import ABC

from alpaca_eval import evaluate as alpaca_farm_evaluate

from evaluation.schemas import OpenEndedTaskBase, GeneratorModelBase


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text: str) -> str:
    """
    extract answer
    """
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    # fallback: 去掉 <think>...</think>
    text_wo_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text_wo_think.strip()


class AlpacaEvalBase(OpenEndedTaskBase, ABC):
    def __init__(self):
        super().__init__()
        self.max_new_tokens, self.temperature, self.top_p = self.prepare_hparams()

    @abc.abstractmethod
    def prepare_hparams(self):
        raise NotImplementedError

    def required_input_fields(self) -> list[str]:
        return ["instruction"]


class AlpacaEval2_0(AlpacaEvalBase):
    def prepare_hparams(self):
        max_new_tokens = 8192
        temperature = 0
        top_p = 1.0
        return max_new_tokens, temperature, top_p

    def _get_eval_data_path(self) -> str:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "alpaca_eval.json")

    def _get_reference_outputs_path(self) -> str:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        gpt4_baseline = os.path.join(current_dir, "alpaca_eval_gpt4_baseline.json")
        if os.path.exists(gpt4_baseline):
            return gpt4_baseline
        return os.path.join(current_dir, "alpaca_eval.json")

    def load(self) -> list[dict]:

        data_path = self._get_eval_data_path()
        print(f"[AlpacaEval] Loading data from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            alpaca_eval_data = json.load(f)
        instructions = [{"instruction": row["instruction"]} for row in alpaca_eval_data]
        print(f"[AlpacaEval] Loaded {len(instructions)} instructions")
        return instructions

    def _evaluate(self, model: GeneratorModelBase) -> tuple[dict, list[dict]]:
        inputs = [{"instruction": row["instruction"]} for row in self.data]
        completions = model.generate_completions(
            inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        assert len(completions) == len(self.data)

        model_id = "_".join(model.model_name_or_path.split("/"))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(current_dir, "cache"), exist_ok=True)
        output_path = os.path.join(current_dir, "results", model_id)
        os.makedirs(output_path, exist_ok=True)

        # 1. Extract <answer>
        print(f"\n[AlpacaEval] Extracting <answer> tags from {len(completions)} responses...")
        model_results = []
        raw_results = []
        extracted_count = 0
        fallback_count = 0
        
        for example, raw_output in zip(self.data, completions):
            answer = extract_answer(raw_output)
            if "<answer>" in raw_output:
                extracted_count += 1
            else:
                fallback_count += 1
            
            model_results.append(
                {
                    "instruction": example["instruction"],
                    "output": answer,
                    "generator": model_id,
                }
            )
            raw_results.append(
                {
                    "instruction": example["instruction"],
                    "raw_output": raw_output,
                    "generator": model_id,
                }
            )
        
        print(f"[AlpacaEval] Successfully extracted {extracted_count} answers, {fallback_count} fallbacks")

        raw_output_path = os.path.join(output_path, "raw_outputs.json")
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        print(f"[AlpacaEval] Saved raw outputs to: {raw_output_path}")

        # 2. GPT-4 baseline
        reference_outputs_path = self._get_reference_outputs_path()
        print(f"[AlpacaEval] Using reference outputs: {reference_outputs_path}")

        # 3. evaluate
        print(f"\n[AlpacaEval] Starting GPT-4 evaluation...")
        annotator_name = "weighted_alpaca_eval_gpt4_turbo"
        df_leaderboard, _ = alpaca_farm_evaluate(
            model_outputs=model_results,
            reference_outputs=reference_outputs_path,  
            annotators_config=annotator_name,
            fn_metric="get_length_controlled_winrate",
            output_path=output_path,
            is_return_instead_of_print=True,
            caching_path=os.path.join(
                current_dir, "cache", "alpaca_eval_annotator_cache.json"
            ),
        )

        # 4.  get annotations
        annotation_path = os.path.join(
            output_path, annotator_name, "annotations.json"
        )
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        
        for annotation, original_data in zip(annotations, self.data):
            annotation["id"] = original_data["id"]

        selected_row = df_leaderboard[df_leaderboard.index == model_id]
        selected_row = selected_row.to_dict(orient="records")[0]
        
        print(f"\n[AlpacaEval] Evaluation complete!")
        print(f"[AlpacaEval] LC Win Rate: {selected_row.get('length_controlled_winrate', 'N/A'):.2f}%")
        print(f"[AlpacaEval] Win Rate: {selected_row.get('win_rate', 'N/A'):.2f}%")
        print(f"[AlpacaEval] Avg Length: {selected_row.get('avg_length', 'N/A'):.0f} tokens")

        return selected_row, annotations
