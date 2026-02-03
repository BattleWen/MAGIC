import enum
import os
from importlib import import_module
from typing import Sequence

import yaml

from evaluation.schemas import TaskBase, JailbreakBase


class EvalMode(enum.Enum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"


def load_evaluation_tasks(
    mode: EvalMode, 
    tasks: list[str],
    classifier_model_name_override: str | None = None,
    attacker_model_path: str | None = None,
    attacker_template: str = "hf",
    use_gpt4_autograder: bool = False,  # ← StrongREJECT GPT-4 autograder参数
    no_extract_answer: bool = False,  # ← 是否不提取<answer>内容
) -> Sequence[TaskBase]:
    """
    Load the evaluators for the given tasks.
    Args:
        mode: Evaluation mode, either generation or classification.
            it will be used to load the tasks from the correct directory.

        tasks: List of tasks to evaluate the model on.
            each task can be xxx or xxx:yyy where xxx is the task name and yyy is the task config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.

    Returns:
        List of evaluators.
    """
    evaluation_tasks = []
    tasks_dir = os.path.dirname(os.path.abspath(__file__))
    for task in tasks:
        # split by ":" to get the task name and the config file name.
        # if the config file name is not provided, it will be assumed as task/default.yaml.
        task_name = task if ":" not in task else task.split(":")[0]
        config_file_name = task.split(":")[-1] if ":" in task else "default"
        task_config_file = os.path.join(tasks_dir, mode.value, task_name, f"{config_file_name}.yaml")

        # import the task module and check if the config file exists.
        module = import_module(f"tasks.{mode.value}.{task_name}")
        if not os.path.exists(task_config_file):
            raise FileNotFoundError(f"Task config file {task_config_file} does not exist.")

        # Load the task config file and create the task object
        with open(task_config_file, "r") as f:
            task_config = yaml.safe_load(f)

        task_class = getattr(module, task_config.pop("task_class"))
        
        # Override classifier_model_name if provided
        if classifier_model_name_override and "classifier_model_name" in task_config:
            task_config["classifier_model_name"] = classifier_model_name_override
            print(f"[TaskLoader] Overriding classifier_model_name to: {classifier_model_name_override}")
        
        # Add attacker parameters if task supports it
        if task_class.__name__ == "HarmbenchAttackerDefender":
            if attacker_model_path:
                task_config["attacker_model_path"] = attacker_model_path
                task_config["attacker_template"] = attacker_template
                print(f"[TaskLoader] Adding attacker_model_path: {attacker_model_path}")
        
        # Add API parameters if task supports it
        if task_class.__name__ == "HarmbenchAttackerDefenderAPI":
            # For API-based tasks, we expect API URLs from task_config
            # No need to inject model paths
            print(f"[TaskLoader] Loading API-based attacker-defender task")
        
        # Add GPT-4 autograder parameter for StrongREJECT tasks
        if task_class.__name__ in ["StrongReject", "StrongRejectSmall"]:
            task_config["use_gpt4_autograder"] = use_gpt4_autograder
            if use_gpt4_autograder:
                print(f"[TaskLoader] Enabling GPT-4 autograder for StrongREJECT")
        
        # Add no_extract_answer parameter for supported tasks
        supported_tasks = [
            "Harmbench", 
            "WildguardTest", 
            "WildJailbreak", 
            "DoAnythingNow", 
            "XSTest", 
            "ORBench"
        ]
        if any(name in task_class.__name__ for name in supported_tasks):
            task_config["no_extract_answer"] = no_extract_answer
            if no_extract_answer:
                print(f"[TaskLoader] Disabling answer extraction for {task_class.__name__}")
        
        evaluation_tasks.append(task_class(**task_config))

    return evaluation_tasks


def wrap_task_with_jailbreak(task: TaskBase, attack: str) -> JailbreakBase:
    """
    Apply the attack to the task.
    Args:
        task: Task to apply the attack.

        attack: Attack to apply.
            similar to task, it can be xxx or xxx:yyy where xxx is the attack name and yyy is the attack config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.

    Returns:
        Jailbreak object.
    """
    module = import_module(f"tasks.jailbreaks.{attack}")
    config_file_name = attack.split(":")[-1] if ":" in attack else "default"
    attack_config_file = os.path.join(
        os.path.dirname(__file__), f"{attack.split(':')[0]}/{config_file_name}.yaml")

    with open(attack_config_file, "r") as f:
        attack_config = yaml.safe_load(f)

    attack_class = getattr(module, attack_config.pop("attack_class"))
    jailbreak_task = attack_class(task, **attack_config)
    return jailbreak_task
