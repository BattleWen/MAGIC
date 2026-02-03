import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    args = argparse.ArgumentParser(
        description="Calculates mean diversity stats across ActorAttack conversations for each behavior"
    )
    args.add_argument(
        "input_file",
        type=str,
        action="store",
        help="Path to JSON file containing conversations",
    )
    parsed_args = args.parse_args()

    print("Loading sentence transformer model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with open(parsed_args.input_file, "r") as actorattack_file:
        actorattack = json.load(actorattack_file)

    def get_plan_embeddings(plans):
        sentences = [f"{plan['actor_name']}\n{plan['relationship']}" for plan in plans]
        return model.encode(sentences)

    def get_convo_embeddings(plans):
        sentences = ["\n".join(plan["multi_turn_queries"]) for plan in plans]
        return model.encode(sentences)

    diversity_scores = {}

    for i in range(0, len(actorattack), 3):
        plans = actorattack[i : i + 2]
        query_id = actorattack[i]["query_id"]
        embeddings = get_convo_embeddings(plans)
        similarity = cosine_similarity(embeddings)
        diversity = 1 - np.mean(similarity)
        diversity_scores[query_id] = diversity.astype(float)
        print(f"Mean diversity for query {query_id}: {diversity}")

    with open("./actor_attack_diversity.json", "w") as means_file:
        json.dump(diversity_scores, means_file, indent=4)

    print("Saved to actor_attack_diversity.json")


if __name__ == "__main__":
    main()
