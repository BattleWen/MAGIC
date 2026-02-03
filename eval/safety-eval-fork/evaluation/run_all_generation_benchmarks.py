import json
import multiprocessing as mp
import os
from typing import Any
from dataclasses import dataclass
from collections import deque

import fire
import torch

from evaluation.utils import maybe_merge_lora_weights_and_save, upload_results_to_hf

def _upload_results_to_hf(task_name: str, primary_metric: str, all_results: dict[str, Any], upload_to_hf, hf_upload_name):
    primary_score = all_results[primary_metric]
    # invert for use in leaderboard
    upload_results_to_hf(
        all_results,
        upload_to_hf,
        hf_upload_name,
        task_name=task_name,
        primary_score=primary_score,
        prepend_timestamp=True,
    )

@dataclass
class GPUTask:
    task_name: str
    gpu_ids: list[int]
    script_format: str

def run(task: GPUTask):
    gpu_ids_str = ",".join(map(str, task.gpu_ids))
    print(f"Running {task.task_name} on GPUs {gpu_ids_str}.")
    command = task.script_format.format(gpu_id=gpu_ids_str, task_name=task.task_name)
    os.system(command)


def allocate_gpus(available_gpus: deque[int], min_gpus_per_task: int) -> list[int]:
    """Allocate minimum required GPUs for a task."""
    if len(available_gpus) < min_gpus_per_task:
        return []
    
    allocated = []
    for _ in range(min_gpus_per_task):
        allocated.append(available_gpus.popleft())
    return allocated


def main(
        model_name_or_path: str,
        model_input_template_path_or_name: str,
        report_output_path: str,
        save_individual_results_path: str,
        min_gpus_per_task: int = 2,  # New parameter
        hf_revision: str = "main",
        hf_upload_name: str | None = None,
        upload_to_hf: str | None = None,
        filter_classifier_name: str | None = None,
        filter_model_override_path: str | None = None,
):
    entrypoint_script_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "eval.py"
    )
    mp.set_start_method("spawn", force=True)

    if os.path.exists(model_name_or_path):
        model_name_or_path = os.path.abspath(model_name_or_path)

    temp_dir_for_merged_weights = maybe_merge_lora_weights_and_save(model_name_or_path)
    if temp_dir_for_merged_weights is not None:
        model_name_or_path = temp_dir_for_merged_weights

    task_names = [
        # # ======= Default =======
        # # "trustllm_jailbreaktrigger",
        # # "toxigen:tiny",
        "harmbench",
        "xstest",
        "wildguardtest",
        "wildjailbreak:benign",
        "wildjailbreak:harmful",
        "do_anything_now",
        "harmbench_precompute",
        "or_bench:hard-1k",
        "or_bench:toxic"

        # # ======= New =======
        # "harmbench:llama_guard3",
        # "wildguardtest:llama_guard3",
        # "wildjailbreak:benign_llama_guard3",
        # "wildjailbreak:harmful_llama_guard3",
        # "do_anything_now:llama_guard3",
        # "xstest:sorry_bench",
    ]

    num_devices = torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")
    
    if min_gpus_per_task > num_devices:
        raise ValueError(f"min_gpus_per_task ({min_gpus_per_task}) cannot be greater than available GPUs ({num_devices})")

    script_format = (
        "CUDA_VISIBLE_DEVICES={gpu_id} \\\n"
        f"python {entrypoint_script_path} generators \\\n"
        f"  --model_name_or_path {model_name_or_path} \\\n"
        f"  --model_input_template_path_or_name {model_input_template_path_or_name} --hf_revision {hf_revision}\\\n"
        "   --tasks {task_name} \\\n"
        f"  --report_output_path {report_output_path}" ".{task_name} \\\n"
        f"  --save_individual_results_path {save_individual_results_path}" ".{task_name} \\\n"
        "   --use_vllm \\\n"
        f"  --filter_classifier_name {filter_classifier_name} \\\n"
        f"  --filter_model_override_path {filter_model_override_path}"
    )

    with mp.Manager() as manager:
        # Initialize task queue and results
        tasks_queue = deque(task_names)
        results = {}
        
        while tasks_queue:
            current_tasks = []
            available_gpus = deque(range(num_devices))
            
            # Allocate GPUs to as many tasks as possible
            while tasks_queue and len(available_gpus) >= min_gpus_per_task:
                task_name = tasks_queue.popleft()
                gpu_ids = allocate_gpus(available_gpus, min_gpus_per_task)
                if gpu_ids:
                    current_tasks.append(GPUTask(
                        task_name=task_name,
                        gpu_ids=gpu_ids,
                        script_format=script_format
                    ))
                else:
                    tasks_queue.append(task_name)
                    break
            
            if not current_tasks:
                raise RuntimeError("Unable to allocate minimum required GPUs for any task")
            
            # Execute current batch of tasks
            with mp.Pool(processes=len(current_tasks)) as pool:
                res = []
                for task in current_tasks:
                    res.append(pool.apply_async(run, (task,)))
                
                # Wait for all processes in current batch to complete
                for task, r in zip(current_tasks, res):
                    try:
                        r.get()
                        results[task.task_name] = True
                    except Exception as e:
                        print(f"Error in {task.task_name}.")
                        print(e)
                        results[task.task_name] = False
                
                pool.close()
                pool.join()

    # Load and merge reports
    report_files = [report_output_path + f".{task_name}" for task_name in task_names]
    save_files = [save_individual_results_path + f".{task_name}" for task_name in task_names]

    all_report = {}
    for report_file in report_files:
        try:
            with open(report_file, "r") as f:
                report = json.load(f)
                all_report.update(report)
        except FileNotFoundError:
            print(f"Report file {report_file} not found.")

    if upload_to_hf is not None:
        wildjailbreak_scores = [
            all_report["wildjailbreak:benign"]["macro ASR"],
            1 - all_report["wildjailbreak:harmful"]["macro ASR"]  # for benign, lower is better
        ]
        primary_scores = [
            1 - all_report["harmbench"]["micro ASR (lower)"],
            1 - all_report["wildguardtest"]["micro harm (lower)"],
            sum(wildjailbreak_scores) / len(wildjailbreak_scores),
            1 - all_report["do_anything_now"]["macro ASR"],
            1 - all_report["trustllm_jailbreaktrigger"]["macro ASR"],
            all_report['xstest']['overall_accuracy']
        ]
        average_score = sum(primary_scores) / len(primary_scores)
        all_report["overall_safety_average"] = average_score
        _upload_results_to_hf(
            "overall_oe_safety_average", "overall_safety_average", all_report, upload_to_hf, hf_upload_name
        )

    with open(report_output_path, "w") as f:
        json.dump(all_report, f)

    # Merge save files
    all_save = []
    for save_file in save_files:
        try:
            with open(save_file, "r") as f:
                save = json.load(f)
                all_save.extend(save)
        except FileNotFoundError:
            print(f"Save file {save_file} not found.")

    with open(save_individual_results_path, "w") as f:
        json.dump(all_save, f)

    if temp_dir_for_merged_weights is not None:
        print(f"Removing temporary directory: {temp_dir_for_merged_weights}")
        os.system(f"rm -rf {temp_dir_for_merged_weights}")


if __name__ == "__main__":
    fire.Fire(main)
