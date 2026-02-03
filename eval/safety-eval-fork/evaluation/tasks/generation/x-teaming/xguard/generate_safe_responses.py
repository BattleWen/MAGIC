import argparse
import concurrent.futures
import json
import logging
import os
import sys
from datetime import datetime

import tiktoken
import tqdm
import yaml

sys.path.append("../")
from agents.base_agent import BaseAgent
from agents.gpt_evaluator import GPTJudge


def truncate_response(response_text: str, max_tokens: int = 512) -> str:
    """Truncates responses to prevent token overflow"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = encoding.encode(response_text)
        if len(tokens) <= max_tokens:
            return response_text
        return encoding.decode(tokens[:max_tokens])
    except Exception as e:
        print(f"Warning: Error in token counting: {e}")
        return response_text


def create_output_directory():
    """Create timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use a relative path or path in user's home directory
    base_output_dir = "./output"
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir):
    """Setup logging to both file and console with ANSI code handling"""

    class NoColorFormatter(logging.Formatter):
        def format(self, record):
            import re

            record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
            return super().format(record)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "execution_log.txt"))
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Setup basic config
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def process_single_convo(behavior, behavior_number, convo, agent, judge, safety_cfg):
    logging.info(f"Processing behavior {behavior_number}")

    convo_fmt = [
        f"User: {turn['attacker']}\nAssistant: {turn['target']}" for turn in convo
    ]
    history_before_jailbreak = "\n\n".join(convo_fmt[:-1])

    formatted_prompt = safety_cfg["prompt"].format(
        harmful_behavior=behavior,
        previous_conversation_history=history_before_jailbreak,
        last_query=convo[-1]["attacker"],
        unsafe_answer=convo[-1]["target"],
    )

    response = agent.call_api(
        messages=[
            {"role": "system", "content": safety_cfg["sysprompt"]},
            {"role": "user", "content": formatted_prompt},
        ],
        temperature=0,
    )

    truncated_response = truncate_response(response)

    logging.info(f"Evaluating new response for behavior {behavior_number}")
    # judge the new response
    score, reason = judge.infer_single(behavior, truncated_response)

    # replace jailbreak response in the history with the clean one
    convo[-1] = {
        "turn": convo[-1]["turn"],
        "phase": convo[-1]["phase"],
        "attacker": convo[-1]["attacker"],
        "target": response,
        "target_truncated": truncated_response,
        "evaluation_score": score,
        "evaluation_reason": reason,
    }
    return convo


def main():
    args = argparse.ArgumentParser(
        description="Generates safety training data by rewriting the final target response of conversations above a certain score."
    )
    args.add_argument(
        "timestamp", action="store", type=str, help="Timestamp of experiment"
    )
    args.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        default="./config.yaml",
        help="Path to YAML config file",
    )
    args.add_argument(
        "-p",
        "--prompt",
        action="store",
        type=str,
        default="./safe_response_prompt.yaml",
        help="Path to YAML containing safe data generation prompt",
    )
    args.add_argument(
        "-t",
        "--threshold_score",
        action="store",
        type=int,
        default=1,
        help="Maximum score that does not need clean response",
    )
    parsed_args = args.parse_args()

    # Load configuration
    with open(parsed_args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(f"../attacks/{parsed_args.timestamp}/all_results.json") as f:
        in_json = json.load(f)

    # Setup
    output_dir = create_output_directory()
    setup_logging(output_dir)
    agent = BaseAgent(config["safety_data_generator"])

    with open(parsed_args.prompt, "r") as f:
        safety_yaml = yaml.safe_load(f)
        safety_cfg = {
            "sysprompt": safety_yaml["prompts"]["system"]["messages"][0]["content"],
            "prompt": safety_yaml["prompts"]["user_message"]["messages"][0]["content"],
        }

    # set up judge
    judge = GPTJudge()

    all_params = []
    for b in in_json["behaviors"].values():
        for s in b["strategies"]:
            if s["conversation"][-1]["evaluation_score"] <= parsed_args.threshold_score:
                continue
            all_params.append(
                {
                    "behavior": b["behavior"]["Behavior"],
                    "behavior_number": b["behavior_number"],
                    "convo": s["conversation"],
                    "safety_cfg": safety_cfg,
                    "agent": agent,
                    "judge": judge,
                }
            )

    all_safe_convos = {}
    # Process each behavior
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_single_convo, **args): args for args in all_params
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            param_dict = futures[future]
            safe_convo = future.result()
            behavior_number = param_dict["behavior_number"]
            if behavior_number not in all_safe_convos:
                all_safe_convos[behavior_number] = {"conversations": []}
            all_safe_convos[behavior_number]["conversations"].append(safe_convo)
            logging.info(f"Saving behavior {behavior_number}")
            # Save results
            with open(os.path.join(output_dir, "safe_convos_scores.json"), "w") as f:
                json.dump(all_safe_convos, f, indent=4, ensure_ascii=False)
    logging.info("Finished")


if __name__ == "__main__":
    main()
