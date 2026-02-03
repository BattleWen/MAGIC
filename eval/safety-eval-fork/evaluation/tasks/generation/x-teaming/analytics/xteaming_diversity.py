import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    args = argparse.ArgumentParser(
        description="Calculates mean diversity stats across X-Teaming conversations for each behavior"
    )
    args.add_argument(
        "input_file",
        type=str,
        action="store",
        help="Path to JSON file containing conversations",
    )
    args.add_argument(
        "-n",
        type=int,
        action="store",
        help="Number of behaviors to randomly sample",
        default=0,
    )
    args.add_argument(
        "-k",
        type=int,
        action="store",
        help="Number of good strategies to consider per behavior",
        default=3,
    )
    parsed_args = args.parse_args()

    print("Loading sentence transformer model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda:0")

    xteam_path = Path(parsed_args.input_file)
    xteam_dir = xteam_path.parent
    output_dir = xteam_dir.joinpath("diversity_stats")
    os.makedirs(output_dir, exist_ok=True)

    with open(xteam_path, "r") as xteam_file:
        xteam_results = json.load(xteam_file)

    def format_convo(strategy):
        return "\n".join([turn["attacker"] for turn in strategy["conversation"]])

    def get_convo_embeddings(strategies):
        sentences = [format_convo(strategy) for strategy in strategies]
        return model.encode(sentences)

    diversity_scores = {}
    if parsed_args.n == 0:
        xteam_behaviors = xteam_results["behaviors"].items()
    else:
        xteam_behaviors = random.sample(
            list(xteam_results["behaviors"].items()), parsed_args.n
        )

    for behavior_number, behavior in tqdm.tqdm(xteam_behaviors):
        good_strategies = list(
            filter(lambda s: s["jailbreak_achieved"], behavior["strategies"])
        )
        if len(good_strategies) < parsed_args.k:
            continue
        embeddings = get_convo_embeddings(good_strategies)
        similarity = cosine_similarity(embeddings)
        diversity = 1 - np.mean(similarity)
        diversity_scores[behavior_number] = diversity.astype(float)

    means_path = output_dir.joinpath("xteaming_diversity.json")
    with open(means_path, "w") as means_file:
        json.dump(diversity_scores, means_file, indent=4)

    print("Saved to xteaming_diversity.json")


if __name__ == "__main__":
    main()
