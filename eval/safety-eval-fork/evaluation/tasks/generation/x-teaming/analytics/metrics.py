import argparse
import concurrent.futures
import json

import numpy as np
import scipy.stats as stats
import tiktoken
import tqdm


def main():
    args = argparse.ArgumentParser(
        description="Prints out a summary of an attack, including statistics like attack success rate (ASR)."
    )
    args.add_argument("timestamp", action="store", type=str)
    args.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print out more detailed statistics.",
    )
    parsed_args = args.parse_args()

    results_path = f"./attacks/{parsed_args.timestamp}/all_results.json"

    with open(results_path, "r") as json_file:
        all_results = json.load(json_file)

        attacker_cfg = all_results["configuration"]["attacker"]
        target_cfg = all_results["configuration"]["target"]

        print("===== Setup of experiment", parsed_args.timestamp, "=====")
        print(f'Attacker:\t\t\t{attacker_cfg["provider"]}:{attacker_cfg["model"]}')
        if "plan_revision" in attacker_cfg:
            print(f'    Plan Revision:\t\t{attacker_cfg["plan_revision"]}')
        if "run_all_strategies" in attacker_cfg:
            print(f'    Run Every Strategy:\t\t{attacker_cfg["run_all_strategies"]}')
        if "strategies_per_behavior" in attacker_cfg:
            print(
                f'    Strategies Per Behavior:\t{attacker_cfg["strategies_per_behavior"]}'
            )
        if "max_turns" in attacker_cfg:
            print(f'    Max Turns:\t\t\t{attacker_cfg["max_turns"]}')
        if "plans_file" in attacker_cfg:
            print(f'    Attack Plan File:\t\t{attacker_cfg["plans_file"]}')
        print(f'Target:\t\t\t\t{target_cfg["provider"]}:{target_cfg["model"]}')
        if "textgrad" in all_results["configuration"]:
            textgrad_cfg = all_results["configuration"]["textgrad"]
            if textgrad_cfg["enabled"]:
                print(
                    f'TextGrad Engine:\t\t{textgrad_cfg["provider"]}:{textgrad_cfg["model"]}'
                )
                if textgrad_cfg["max_turns_per_phase"]:
                    print(
                        f'    Max Turns Per Phase:\t{textgrad_cfg["max_turns_per_phase"]}'
                    )

        print("\n===== Metrics for experiment", parsed_args.timestamp, "=====")
        metrics = dict(
            num_strategies=0,
            successful_strategies=set(),
            all_sets=set(),
            successful_sets=set(),
            num_behaviors=len(all_results["behaviors"]),
            successful_behaviors=set(),
            # histograms
            turns_taken_histo={i: 0 for i in range(1, 11)},
            plans_taken_histo={i: 0 for i in range(1, 51)},
            textgrad_histo={i: 0 for i in range(0, 10)},
            num_tokens=[],
        )

        def count_tokens(response_text):
            encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
            tokens = encoding.encode(response_text)
            return len(tokens)

        def format_full_conversation(convo):
            convo_strs = []
            latest_phase = 0
            for turn in convo:
                if turn["phase"] == latest_phase:
                    # don't count retries of phases in the context
                    convo_strs.pop()
                latest_phase = turn["phase"]
                convo_strs.append(
                    f"<|im_start|>user<|im_sep|>{turn['attacker']}<|im_end|><|im_start|>assistant<|im_sep|>{turn['target']}<|im_end|>"
                )
            return "".join(convo_strs)

        semantic_categories = {}

        def analyze_single_behavior(b):
            metrics["num_strategies"] += len(b["strategies"])

            category = b["behavior"].get("SemanticCategory", "unknown")
            if category not in semantic_categories:
                semantic_categories[category] = {"successful": set(), "total": set()}

            semantic_categories[category]["total"].add(b["behavior_number"])

            for s in b["strategies"]:

                # there are multiple strategies within a set, and multiple sets per behavior
                set_id = f'b{b["behavior_number"]}_s{s["set_number"]}'
                metrics["all_sets"].add(set_id)

                if s["jailbreak_achieved"]:
                    # count number of tokens per plan
                    full_convo = format_full_conversation(s["conversation"])
                    metrics["num_tokens"].append(count_tokens(full_convo))
                    # count number of turns taken per plan
                    strategy_id = f'{set_id}-{s["strategy_number"]}'
                    metrics["successful_behaviors"].add(b["behavior_number"])
                    metrics["turns_taken_histo"][len(s["conversation"])] += 1
                    num_textgrad_turns = sum(
                        [
                            turn.get("loss", None) is not None
                            for turn in s["conversation"]
                        ]
                    )
                    metrics["textgrad_histo"][num_textgrad_turns] += 1
                    semantic_categories[category]["successful"].add(
                        b["behavior_number"]
                    )
                    metrics["successful_sets"].add(set_id)
                    metrics["successful_strategies"].add(strategy_id)

            # count number of plans taken per behavior
            if b["behavior_number"] in metrics["successful_behaviors"]:
                metrics["plans_taken_histo"][len(b["strategies"])] += 1

        all_behaviors = all_results["behaviors"].values()

        # analyze behaviors in parallel, because long runs may have thousands of them
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(analyze_single_behavior, b=behavior): behavior
                for behavior in all_behaviors
            }
            iterator = concurrent.futures.as_completed(futures)
            if len(futures) >= 1000:
                iterator = tqdm.tqdm(iterator, total=len(futures))
            for future in iterator:
                pass

        print("Behavior ASR:")
        if parsed_args.verbose:
            print(
                f"Set ASR     \t{len(metrics['successful_sets'])}/{len(metrics['all_sets'])}\t{100.0 * len(metrics['successful_sets']) / len(metrics['all_sets']):.1f}%"
            )
            print(
                f"Strategy ASR\t{len(metrics['successful_strategies'])}/{metrics['num_strategies']}\t{100.0 * len(metrics['successful_strategies']) / metrics['num_strategies']:.1f}%"
            )
            print("Category")
            for label in sorted(semantic_categories.keys()):
                collection = semantic_categories[label]
                print(
                    f"    {label:<30}\t{len(collection['successful'])}/{len(collection['total'])}\t{100.0 * len(collection['successful']) / len(collection['total']):.1f}%"
                )
        print(
            f"Total\t\t\t\t\t{len(metrics['successful_behaviors'])}/{metrics['num_behaviors']}\t{100.0 * len(metrics['successful_behaviors']) / metrics['num_behaviors']:.1f}%"
        )

        def margin_of_error(x):
            std_dev = np.std(x, ddof=1)
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            return z_score * (std_dev / np.sqrt(len(x)))

        def process_histogram_dict(histo):
            """
            Calculates statistics and prints out histogram data from a dictionary.
            """
            data = []
            # flattens the histogram into a lits
            for value, frequency in histo.items():
                data.extend([value] * frequency)
            for i, count in histo.items():
                if parsed_args.verbose:
                    print(f"    {i:>2}\t{count}/{len(data)}")
            avg = np.mean(data)
            error = margin_of_error(data) if len(data) > 1 else 0
            print(f"Average\t\t\t\t\t{avg:.3f} ± {error:.3f}")

        print("Number of Turns Taken:")
        process_histogram_dict(metrics["turns_taken_histo"])

        print("Number of Plans Used:")
        process_histogram_dict(metrics["plans_taken_histo"])

        print("Number of TextGrad Used:")
        process_histogram_dict(metrics["textgrad_histo"])

        print("Number of Tokens Used:")
        print(
            f"Average\t\t\t\t\t{np.average(metrics['num_tokens']):.3f} ± {margin_of_error(metrics['num_tokens']):.3f}"
        )


if __name__ == "__main__":
    main()
