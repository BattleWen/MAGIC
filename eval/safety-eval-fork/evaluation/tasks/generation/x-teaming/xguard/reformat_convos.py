import argparse
import json


def reformat_convo(convo):
    new_convo = []
    for turn in convo:
        new_convo.append({"from": "human", "value": turn["attacker"]})
        new_convo.append({"from": "gpt", "value": turn["target"]})
    return new_convo


def main():
    args = argparse.ArgumentParser(
        description="Converts a JSON of conversations from our format into ShareGPT format for supervised fine-tuning"
    )
    args.add_argument(
        "in_path",
        action="store",
        type=str,
        help="Path to a JSON file containing conversations",
    )
    args.add_argument(
        "out_path", action="store", type=str, help="Path to store output JSON"
    )
    args.add_argument(
        "-t",
        "--threshold_score",
        action="store",
        type=int,
        default=1,
        help="Maximum score that should be included",
    )

    cfg = args.parse_args()

    print(f"Reading input JSON: {cfg.in_path}")

    with open(cfg.in_path, "r") as in_file:
        in_json = json.load(in_file)

    out_plans = []

    if cfg.in_path.endswith("safe_convos_scores.json"):
        # json came from generate_safe_responses.py
        for b in in_json.values():
            for c in b["conversations"]:
                if c[-1]["evaluation_score"] <= cfg.threshold_score:
                    out_plans.append({"conversations": reformat_convo(c)})
    else:
        # assume this json is an all_results.json
        for b in in_json["behaviors"].values():
            for s in b["strategies"]:
                if s["conversation"][-1]["evaluation_score"] <= cfg.threshold_score:
                    out_plans.append(
                        {"conversations": reformat_convo(s["conversation"])}
                    )

    print(
        f"Writing {len(out_plans)} strategies with score <= {cfg.threshold_score} to {cfg.out_path}"
    )

    with open(cfg.out_path, "w") as out_file:
        json.dump(out_plans, out_file, indent=4, ensure_ascii=False)

    print("Finished")


if __name__ == "__main__":
    main()
