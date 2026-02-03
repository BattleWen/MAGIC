import argparse
import concurrent.futures
import json
import logging
import os
import sys
from datetime import datetime
from pprint import pprint

import pandas as pd
import tqdm
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

sys.path.append("../")
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


def load_and_format_prompts(behavior, previous_responses=None):
    """Load and format prompts based on set number"""
    with open("./plan_generation_prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    system_prompt = prompts["prompts"]["system"]["messages"][0]["content"]
    user_prompt = prompts["prompts"]["user_message1"]["messages"][0]["content"]
    formatted_user_prompt = user_prompt.replace("{target_behavior}", behavior)
    return system_prompt, formatted_user_prompt


def create_output_directory(base_output_dir):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def generate_strategies(agent, messages, temperature):
    """Generate strategies for a single set"""
    response = agent.call_api(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    # Parse the response string into a Python dictionary
    parsed_response = json.loads(response)
    assert len(parsed_response) == 5

    logging.info(response)  # Keep original logging

    return parsed_response  # Return parsed dictionary instead of raw string


def process_single_behavior(i, row, agent, temperature):
    behavior = row["Behavior"]
    behavior_id = row["id"]
    logging.info(f"\n{'='*50}")
    logging.info(f"Processing Behavior {i} (ID: {behavior_id}):")
    logging.info(f"Behavior: {behavior}")
    logging.info(f"{'='*50}")

    all_messages = []

    # Initialize behavior data
    all_responses = []
    behavior_details = {
        k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()
    }
    behavior_data = {
        "behavior_number": i,
        "behavior_details": behavior_details,
        "attack_strategies": all_responses,
    }

    system_prompt, formatted_user_prompt = load_and_format_prompts(
        behavior=behavior, previous_responses=all_responses
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt},
    ]

    logging.info("Messages for this behavior")
    logging.info(pprint(messages, indent=2))

    message_data = {"behavior_index": i, "behavior": behavior, "messages": messages}
    all_messages.append(message_data)

    response = generate_strategies(
        agent=agent,
        messages=messages,
        temperature=temperature,
    )

    all_responses.append(response)

    return behavior_data, all_messages


def main():
    args = argparse.ArgumentParser(
        description="Tweaked version of attack plan generation script for XGuard-Train seed data"
    )
    args.add_argument("-c", "--config", action="store", default="./config.yaml")
    parsed_args = args.parse_args()

    # Load configuration
    logging.info(f"Reading config from: {parsed_args.config}")
    with open(parsed_args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    output_dir = create_output_directory(
        config["attack_plan_generator"]["attack_plan_generation_dir"]
    )
    setup_logging(output_dir)
    agent = BaseAgent(config["attack_plan_generator"])
    logging.info(
        f"Generating attack plans from csv file: {config['attack_plan_generator']['behavior_path']}"
    )
    df = pd.read_csv(config["attack_plan_generator"]["behavior_path"])

    all_behaviors_data = []
    all_messages = []

    all_params = []
    for i, row in df.iterrows():
        all_params.append(
            {
                "i": i,
                "row": row,
                "agent": agent,
                "temperature": config["attack_plan_generator"]["temperature"],
            }
        )

    # Process each behavior
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_single_behavior, **args): args
            for args in all_params
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            behavior_result, messages_result = future.result()
            all_behaviors_data.append(behavior_result)
            all_messages += messages_result
            # Save results
            with open(os.path.join(output_dir, "attack_plans.json"), "w") as f:
                json.dump(all_behaviors_data, f, indent=4, ensure_ascii=False)

            with open(os.path.join(output_dir, "all_messages.json"), "w") as f:
                json.dump(all_messages, f, indent=4, ensure_ascii=False)
    logging.info("Finished")


if __name__ == "__main__":
    main()
