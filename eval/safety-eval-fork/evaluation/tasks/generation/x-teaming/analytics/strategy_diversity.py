import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def flatten_strategies(input_file, output_file, n=0):
    """
    Flattens the strategy JSON structure by removing set-level grouping.
    All strategies will be placed at the same level with unique IDs.
    """
    # Read the JSON file
    with open(input_file, "r") as f:
        all_plans = json.load(f)

    if n > 0:
        all_plans = random.sample(all_plans, n)

    all_flattened = []
    for data in all_plans:
        # Create new flattened structure
        flattened_data = {
            "behavior_number": data.get("behavior_number"),
            "behavior_details": data.get("behavior_details"),
            "attack_strategies": {},
        }

        # Flatten the strategies
        strategy_counter = 1
        for set_name, set_strategies in data["attack_strategies"].items():
            # Add each strategy to the flattened dict with a new sequential number
            for _, strategy in set_strategies.items():
                strategy_id = f"strategy_{strategy_counter}"
                flattened_data["attack_strategies"][strategy_id] = strategy
                print(strategy_counter, strategy["persona"])
                strategy_counter += 1

        all_flattened.append(flattened_data)

    # Write the flattened JSON to file
    with open(output_file, "w") as f:
        json.dump(all_flattened, f, indent=4)

    print(f"Flattened {len(all_flattened)} behaviors")
    return all_flattened


def add_embeddings_to_strategies(input_file, output_file):
    print("Loading embedding model")
    # Load the embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda:0")

    # Read the JSON file
    with open(input_file, "r") as f:
        all_flattened = json.load(f)

    print("Generating embeddings")
    for data in tqdm.tqdm(all_flattened):
        # Collect all this behavior's strategy texts for batch embedding
        strategy_texts = []
        strategy_ids = []

        for strategy_id, strategy in data["attack_strategies"].items():
            # Concat the essential fields from the strategy dict into a string
            strategy_text = (
                f"{strategy['persona']}\n{strategy['context']}\n{strategy['approach']}"
            )
            strategy_texts.append(strategy_text)
            strategy_ids.append(strategy_id)

        # Generate embeddings for all texts at once
        embeddings = model.encode(strategy_texts)

        # Add embeddings to the strategies
        for strategy_id, embedding in zip(strategy_ids, embeddings):
            data["attack_strategies"][strategy_id]["embedding"] = embedding.tolist()

    # Write the modified JSON back to file
    with open(output_file, "w") as f:
        json.dump(all_flattened, f, indent=4)


def compute_diversity_matrix(data):
    # Extract strategy IDs and embeddings
    strategies = data["attack_strategies"]
    strategy_ids = list(strategies.keys())
    embeddings = np.array([strategies[sid]["embedding"] for sid in strategy_ids])

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    diversity_matrix = 1 - similarity_matrix

    return diversity_matrix, strategy_ids


def plot_similarity_matrix(data, output_image="similarity_matrix.png"):
    diversity_matrix, strategy_ids = compute_diversity_matrix(data)

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        diversity_matrix,
        xticklabels=strategy_ids,
        yticklabels=strategy_ids,
        cmap="RdYlBu",  # Red-Yellow-Blue colormap
        vmin=0,
        vmax=1,
        annot=True,  # Show numerical values
        fmt=".2f",  # Format to 2 decimal places
        square=True,
    )

    plt.title("Strategy Diversity Matrix")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()


def prune_strategies(
    input_file, output_file_fmt=None, k=None, similarity_threshold=None
):
    """
    Prune strategies for plotting using Maximum Dissimilarity Selection.
    Either keeps k most diverse strategies or all strategies below similarity threshold.

    Args:
        input_file: Path to JSON file with strategy embeddings
        output_file_fmt: Optional path format string to save pruned strategies. If None, returns dict instead.
        k: Optional number of strategies to keep. If provided, keeps exactly k most diverse strategies.
        similarity_threshold: Optional maximum allowed similarity (0.0 to 1.0).
                            If provided, keeps all strategies below this similarity threshold.

    Note: Provide either k or similarity_threshold, not both.
    """
    if k is not None and similarity_threshold is not None:
        raise ValueError("Provide either k or similarity_threshold, not both")

    # Read the JSON file
    with open(input_file, "r") as f:
        all_flattened = json.load(f)

    all_pruned = []
    for data in all_flattened:
        strategies = data["attack_strategies"]
        strategy_ids = list(strategies.keys())

        if k is not None and k >= len(strategy_ids):
            print(
                f"Warning: k ({k}) is larger than or equal to number of strategies ({len(strategy_ids)})"
            )
            return data

        embeddings = np.array([strategies[sid]["embedding"] for sid in strategy_ids])
        similarity_matrix = cosine_similarity(embeddings)

        # Initialize with the first strategy
        kept_indices = [0]
        kept_strategy_ids = [strategy_ids[0]]

        # Iteratively add most dissimilar strategies
        while True:
            # Calculate maximum similarity to any kept strategy for each candidate
            max_similarities = np.max(similarity_matrix[kept_indices][:, :], axis=0)

            # Find candidate indices (all indices not yet kept)
            candidate_indices = [
                i for i in range(len(strategy_ids)) if i not in kept_indices
            ]
            if not candidate_indices:
                break

            # Find the strategy with lowest maximum similarity to kept strategies
            similarities = [max_similarities[i] for i in candidate_indices]
            best_candidate_idx = candidate_indices[np.argmin(similarities)]

            # Check stopping condition based on k or similarity_threshold
            if k is not None:
                if len(kept_indices) >= k:
                    break
                kept_indices.append(best_candidate_idx)
                kept_strategy_ids.append(strategy_ids[best_candidate_idx])
            else:  # using similarity_threshold
                if max_similarities[best_candidate_idx] >= similarity_threshold:
                    break
                kept_indices.append(best_candidate_idx)
                kept_strategy_ids.append(strategy_ids[best_candidate_idx])

        # Create new data with only kept strategies and preserve metadata
        pruned_data = {
            "behavior_number": data.get("behavior_number"),
            "behavior_details": data.get("behavior_details"),
            "attack_strategies": {sid: strategies[sid] for sid in kept_strategy_ids},
        }

        all_pruned.append(pruned_data)

        if output_file_fmt:
            with open(output_file_fmt.format(data.get("behavior_number")), "w") as f:
                json.dump(pruned_data, f, indent=4)

        print(
            f"Behavior {data['behavior_number']}: Kept {len(kept_strategy_ids)} strategies out of {len(strategy_ids)}"
        )

    return all_pruned


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Generates heatmap visualization of plan diversity"
    )
    args.add_argument(
        "input_file",
        type=str,
        action="store",
        help="Path to JSON file containing attack plans",
    )
    args.add_argument(
        "-n",
        type=int,
        action="store",
        help="Number of behaviors to randomly sample (0 = all)",
        default=0,
    )
    args.add_argument(
        "-k",
        type=int,
        action="store",
        help="Number of strategies to keep in max dissimilarity selection",
        default=10,
    )
    parsed_args = args.parse_args()

    input_file = Path(parsed_args.input_file)
    input_dir = input_file.parent
    output_dir = input_dir.joinpath("diversity_stats")
    os.makedirs(output_dir, exist_ok=True)

    # Flatten strategies
    flattened_path = output_dir.joinpath("flattened.json")
    flattened_data = flatten_strategies(input_file, flattened_path, parsed_args.n)

    # Add embeddings to strategies
    embedding_path = output_dir.joinpath("embeddings.json")
    add_embeddings_to_strategies(flattened_path, embedding_path)

    # Prune strategies
    pruned_file = output_dir.joinpath("pruned_{}.json")

    # Example usage with k:
    pruned_data = prune_strategies(
        embedding_path, output_file_fmt=str(pruned_file), k=parsed_args.k
    )

    # Visualize similarity matrix of pruned strategies
    print("Writing heatmaps...")
    for data in tqdm.tqdm(pruned_data):
        pruned_plot_path = output_dir.joinpath(
            f"dissimilarity_matrix_{data['behavior_number']}.png"
        )
        plot_similarity_matrix(data, pruned_plot_path)

    # Compute mean diversity of all behaviors' strategies
    means = {}
    with open(embedding_path, "r") as embedding_file:
        data = json.load(embedding_file)
        for behavior in data:
            diversity_matrix, _ = compute_diversity_matrix(behavior)
            means[behavior["behavior_number"]] = np.mean(diversity_matrix)

    means_path = output_dir.joinpath("mean_diversity.json")
    with open(means_path, "w") as means_file:
        json.dump(means, means_file, indent=4)

    print("Saved mean_diversity.json")
    print("Finished")
