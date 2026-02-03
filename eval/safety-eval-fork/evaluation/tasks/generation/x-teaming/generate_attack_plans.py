import argparse
import concurrent.futures
import json
import logging
import os
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import tqdm
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

from agents.base_agent import BaseAgent


def setup_logging(output_dir):
    """Setup logging to both file and console with ANSI code handling"""

    class NoColorFormatter(logging.Formatter):
        def format(self, record):
            import re

            record.msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", str(record.msg))
            return super().format(record)

    file_handler = logging.FileHandler(os.path.join(output_dir, "generation_log.txt"))
    file_handler.setFormatter(
        NoColorFormatter("%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def load_datasets(csv_path):
    """Load and filter dataset"""
    np.random.seed(42)
    df = pd.read_csv(csv_path)
    filtered_df = df[df["FunctionalCategory"] == "standard"]

    dedupe_columns = []
    if "BehaviorID" in filtered_df.columns:
        dedupe_columns.append("BehaviorID")
    if "Behavior" in filtered_df.columns:
        dedupe_columns.append("Behavior")
    if dedupe_columns:
        filtered_df = filtered_df.drop_duplicates(subset=dedupe_columns, keep="first")

    return filtered_df.sample(frac=1).reset_index(drop=True)


def load_and_format_prompts(behavior, set_number, previous_responses=None):
    """Load and format prompts based on set number"""
    with open("config/prompts/plan_generation_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]

    if set_number == 1:
        user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
        formatted_user_prompt = user_prompt.replace("{target_behavior}", behavior)
    else:
        user_prompt = prompts["prompts"]["user_message2"]["messages"][0]["content"]
        formatted_user_prompt = user_prompt.replace("{target_behavior}", behavior)

        strategies_text = ""
        for set_name, response in previous_responses.items():
            strategies_text += f"\n{set_name}:\n{response}\n"
        formatted_user_prompt = formatted_user_prompt.replace(
            "{previously_generated_strategies}", strategies_text
        )

    return system_prompt, formatted_user_prompt


def create_output_directory(base_output_dir):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def generate_strategies(agent, messages, set_num, temperature):
    """Generate strategies for a single set"""
    response = agent.call_api(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    # Parse the response string into a Python dictionary
    parsed_response = json.loads(response)
    assert len(parsed_response) == 10

    logging.info(f"\nSet {set_num} Generated Strategies:")
    logging.info(response)  # Keep original logging

    return parsed_response  # Return parsed dictionary instead of raw string


def process_single_behavior(i, row, agent, temperature, num_sets=5):
    behavior = row["Behavior"]
    behavior_id = row["BehaviorID"]
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Behavior {i} (ID: {behavior_id}):")
    logging.info(f"Behavior: {behavior}")
    logging.info(f"{'='*50}")

    all_messages = []

    # Initialize behavior data
    all_responses = {}
    behavior_details = {
        k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
    }
    behavior_data = {
        "behavior_number": i,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }

    # Generate strategies for each set
    for set_num in range(1, num_sets + 1):
        logging.info(f"\nGenerating Set {set_num}:")

        system_prompt, formatted_user_prompt = load_and_format_prompts(
            behavior=behavior, set_number=set_num, previous_responses=all_responses
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        logging.info(f"Messages for set {set_num}")
        logging.info(pprint(messages, indent=2))

        message_data = {
            "behavior_index": i,
            "behavior": behavior,
            "set_number": set_num,
            "messages": messages,
        }
        all_messages.append(message_data)

        response = generate_strategies(
            agent=agent,
            messages=messages,
            set_num=set_num,
            temperature=temperature,
        )

        all_responses[f"Set_{set_num}"] = response

    return behavior_data, all_messages


def main():
    args = argparse.ArgumentParser(
        description="Generates multi-turn jailbreak attack strategies for X-Teaming."
    )
    args.add_argument(
        "-c", "--config", action="store", type=str, default="./config/config.yaml"
    )
    parsed_args = args.parse_args()

    config_path = parsed_args.config

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    behavior_path = config["attack_plan_generator"]["behavior_path"]
    if not os.path.isabs(behavior_path):
        behavior_path = os.path.normpath(os.path.join(base_dir, behavior_path))

    attack_plan_generation_dir = config["attack_plan_generator"]["attack_plan_generation_dir"]
    if not os.path.isabs(attack_plan_generation_dir):
        attack_plan_generation_dir = os.path.normpath(
            os.path.join(base_dir, attack_plan_generation_dir)
        )

    # Setup
    output_dir = create_output_directory(
        attack_plan_generation_dir
    )
    setup_logging(output_dir)
    agent = BaseAgent(config["attack_plan_generator"])
    df = load_datasets(behavior_path)

    all_behaviors_data = []
    all_messages = []

    target_success_count = config["attack_plan_generator"]["num_behaviors"]
    temperature = config["attack_plan_generator"]["temperature"]
    num_sets = config["attack_plan_generator"].get("sets_per_behavior", 5)
    max_workers = config["attack_plan_generator"].get("max_workers", 10)

    pbar = tqdm.tqdm(total=target_success_count)
    pending: dict[concurrent.futures.Future, dict] = {}
    rows_iter = df.iterrows()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while len(all_behaviors_data) < target_success_count:
            while len(pending) < max_workers:
                try:
                    i, row = next(rows_iter)
                except StopIteration:
                    break

                params = {
                    "i": i,
                    "row": row,
                    "agent": agent,
                    "temperature": temperature,
                    "num_sets": num_sets,
                }
                future = executor.submit(process_single_behavior, **params)
                pending[future] = params

            if not pending:
                break

            done, _ = concurrent.futures.wait(
                pending.keys(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                params = pending.pop(future)
                try:
                    behavior_result, messages_result = future.result()
                    all_behaviors_data.append(behavior_result)
                    all_messages.extend(messages_result)
                    pbar.update(1)

                    with open(os.path.join(output_dir, "attack_plans.json"), "w") as f:
                        json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)

                    with open(os.path.join(output_dir, "all_messages.json"), "w") as f:
                        json.dump(all_messages, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    behavior_id = params["row"]["BehaviorID"]
                    logging.error(
                        f"Behavior {behavior_id} generated an exception",
                        exc_info=e,
                    )

    pbar.close()
    behaviors_csv_path = os.path.join(output_dir, f"behaviors_{len(all_behaviors_data)}.csv")
    pd.DataFrame([b["behavior_details"] for b in all_behaviors_data]).to_csv(
        behaviors_csv_path, index=False, encoding="utf-8-sig"
    )
    logging.info(f"Finished. Generated {len(all_behaviors_data)} attack plans.")


if __name__ == "__main__":
    main()
