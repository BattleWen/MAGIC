"""
Analyze Pattern Diversity - Compare SFT vs RL Attack Strategy Diversity
Analyzes primary_strategy field and generates visualizations.
"""

import json
import os
import glob
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# Experiment presets
EXPERIMENT_RL_DIRS = {
    "attacker-base": "",
    "attacker-only": "",
    "attacker-sft": "",
}

DEFAULT_SFT_DIR = ""

BASELINE_DIRS = {
    "attacker-base": "",
    "attacker-only": DEFAULT_SFT_DIR,
    "attacker-sft": DEFAULT_SFT_DIR,
}

BASELINE_LABELS = {
    "attacker-base": "Base",
    "attacker-only": "SFT",
    "attacker-sft": "SFT",
}

# NOTE (2025-01-16): Synced to prompts_grained.py (20 subcategories, 0-19).
STRATEGY_NAMES = {
    "0": "Role-playing",
    "1": "Simulated decision-making scenarios",
    "2": "Academic/Educational pretext",
    "3": "Concealment of key information",
    "4": "Causation confusion",
    "5": "False analogies & Fact distortion",
    "6": "Concept substitution",
    "7": "False premise assumption",
    "8": "Legitimacy pretext",
    "9": "Ignore previous instructions",
    "10": "Step-by-step questioning",
    "11": "Multi-condition stacking",
    "12": "Complex logic nesting",
    "13": "Encoding rewriting",
    "14": "Templated output format",
    "15": "Code by pseudonym",
    "16": "Structured response",
    "17": "Surrogate modality",
    "18": "Language translation",
    "19": "None of these"
}


def get_step_sort_key(key: str) -> int:
    match = re.search(r"step_(\d+)", key)
    if match:
        return int(match.group(1))
    return 0


def extract_primary_strategies(file_path: str) -> List[str]:
    """Extract all primary_strategy values from a JSONL file"""
    strategies = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    classification = record.get("classification", {})
                    primary = classification.get("primary_strategy")
                    if primary is not None:
                        strategies.append(str(primary))
            except Exception as e:
                continue
    return strategies


def analyze_strategies(strategies: List[str]) -> Dict[str, Any]:
    """Analyze strategy distribution"""
    total = len(strategies)
    if total == 0:
        return {
            "total": 0,
            "single_count": 0,
            "hybrid_count": 0,
            "single_ratio": 0,
            "hybrid_ratio": 0,
            "individual_distribution": {}
        }
    
    # Count single vs hybrid
    single_count = sum(1 for s in strategies if "_" not in s)
    hybrid_count = total - single_count
    
    # Split and count individual strategies
    individual_counts = Counter()
    for strategy in strategies:
        parts = strategy.split("_")
        for part in parts:
            if part:
                individual_counts[part] += 1
    
    return {
        "total": total,
        "single_count": single_count,
        "hybrid_count": hybrid_count,
        "single_ratio": single_count / total if total > 0 else 0,
        "hybrid_ratio": hybrid_count / total if total > 0 else 0,
        "individual_distribution": dict(individual_counts)
    }


def analyze_baseline_data(baseline_dir: str, baseline_label: str) -> Dict[str, Any]:
    """Analyze all baseline data as a whole"""
    print(f"Analyzing {baseline_label} data...")
    
    # Find all classified files
    pattern = os.path.join(baseline_dir, "classified_*.jsonl")
    files = sorted(glob.glob(pattern))
    
    all_strategies = []
    for file_path in files:
        strategies = extract_primary_strategies(file_path)
        all_strategies.extend(strategies)
        print(f"  {Path(file_path).name}: {len(strategies)} strategies")
    
    results = analyze_strategies(all_strategies)
    print(f"  Total {baseline_label} strategies: {results['total']}")
    print(f"  Single: {results['single_count']} ({results['single_ratio']*100:.2f}%)")
    print(f"  Hybrid: {results['hybrid_count']} ({results['hybrid_ratio']*100:.2f}%)")
    
    return results


def analyze_rl_data(rl_dir: str) -> Dict[str, Dict[str, Any]]:
    """Analyze RL data in fixed step ranges"""
    print("\nAnalyzing RL data...")
    
    # Find all classified files
    pattern = os.path.join(rl_dir, "classified_extracted_step_*.jsonl")
    files = sorted(glob.glob(pattern))
    
    step_groups = defaultdict(list)
    for file_path in files:
        filename = Path(file_path).name
        match = re.search(r'step_(\d+)', filename)
        if match:
            step = int(match.group(1))
            
            # Group by fixed intervals
            group_key = None
            if 20 <= step <= 60:
                group_key = "step_1-60"
            elif 140 <= step <= 180:
                group_key = "step_121-180"
            elif 260 <= step <= 300:
                group_key = "step_241-300"
            
            if group_key:
                step_groups[group_key].append(file_path)
    
    # Analyze each group
    results = {}
    for group_key in sorted(step_groups.keys(), key=get_step_sort_key):
        print(f"\n  Analyzing {group_key}...")
        all_strategies = []
        for file_path in step_groups[group_key]:
            strategies = extract_primary_strategies(file_path)
            all_strategies.extend(strategies)
            print(f"    {Path(file_path).name}: {len(strategies)} strategies")
        
        group_results = analyze_strategies(all_strategies)
        results[group_key] = group_results
        print(f"    Total: {group_results['total']}")
        print(f"    Single: {group_results['single_count']} ({group_results['single_ratio']*100:.2f}%)")
        print(f"    Hybrid: {group_results['hybrid_count']} ({group_results['hybrid_ratio']*100:.2f}%)")
    
    return results


def analyze_rl_data_five_stage(rl_dir: str) -> Dict[str, Dict[str, Any]]:
    """Analyze RL data in fixed 5-stage step ranges (for single_vs_hybrid plot)."""
    print("\nAnalyzing RL data (5-stage for single_vs_hybrid)...")
    
    pattern = os.path.join(rl_dir, "classified_extracted_step_*.jsonl")
    files = sorted(glob.glob(pattern))
    
    step_groups = defaultdict(list)
    for file_path in files:
        filename = Path(file_path).name
        match = re.search(r'step_(\d+)', filename)
        if match:
            step = int(match.group(1))
            
            group_key = None
            if 20 <= step <= 60:
                group_key = "step_20-60"
            elif 80 <= step <= 120:
                group_key = "step_80-120"
            elif 140 <= step <= 180:
                group_key = "step_140-180"
            elif 200 <= step <= 240:
                group_key = "step_200-240"
            elif 260 <= step <= 300:
                group_key = "step_260-300"
            
            if group_key:
                step_groups[group_key].append(file_path)
    
    results = {}
    for group_key in sorted(step_groups.keys(), key=get_step_sort_key):
        print(f"\n  Analyzing {group_key}...")
        all_strategies = []
        for file_path in step_groups[group_key]:
            strategies = extract_primary_strategies(file_path)
            all_strategies.extend(strategies)
            print(f"    {Path(file_path).name}: {len(strategies)} strategies")
        
        group_results = analyze_strategies(all_strategies)
        results[group_key] = group_results
        print(f"    Total: {group_results['total']}")
        print(f"    Single: {group_results['single_count']} ({group_results['single_ratio']*100:.2f}%)")
        print(f"    Hybrid: {group_results['hybrid_count']} ({group_results['hybrid_ratio']*100:.2f}%)")
    
    return results


def analyze_rl_data_fifteen_stage(rl_dir: str) -> Dict[str, Dict[str, Any]]:
    """Analyze RL data in fixed 15-stage step ranges (for single_vs_hybrid plot)."""
    print("\nAnalyzing RL data (15-stage for single_vs_hybrid)...")
    
    target_steps = list(range(20, 301, 20))
    step_groups = {f"step_{step}": [] for step in target_steps}
    
    pattern = os.path.join(rl_dir, "classified_extracted_step_*.jsonl")
    files = sorted(glob.glob(pattern))
    
    for file_path in files:
        filename = Path(file_path).name
        match = re.search(r'step_(\d+)', filename)
        if match:
            step = int(match.group(1))
            if step in target_steps:
                step_groups[f"step_{step}"].append(file_path)
    
    results = {}
    for step in target_steps:
        group_key = f"step_{step}"
        print(f"\n  Analyzing {group_key}...")
        all_strategies = []
        for file_path in step_groups[group_key]:
            strategies = extract_primary_strategies(file_path)
            all_strategies.extend(strategies)
            print(f"    {Path(file_path).name}: {len(strategies)} strategies")
        
        group_results = analyze_strategies(all_strategies)
        results[group_key] = group_results
        print(f"    Total: {group_results['total']}")
        print(f"    Single: {group_results['single_count']} ({group_results['single_ratio']*100:.2f}%)")
        print(f"    Hybrid: {group_results['hybrid_count']} ({group_results['hybrid_ratio']*100:.2f}%)")
    
    return results


def plot_single_vs_hybrid(
    sft_results: Optional[Dict],
    rl_results: Dict,
    output_dir: str,
    exp_label: str,
    baseline_label: str,
    include_sft: bool = True
):
    print("\nGenerating single vs hybrid ratio plot...")
    
    rl_keys = sorted(rl_results.keys(), key=get_step_sort_key)
    if include_sft:
        if sft_results is None:
            raise ValueError("sft_results is required when include_sft is True.")
        labels = [baseline_label] + rl_keys
        hybrid_ratios = [sft_results["hybrid_ratio"] * 100]
        for key in rl_keys:
            hybrid_ratios.append(rl_results[key]["hybrid_ratio"] * 100)
        stage_note = f"{baseline_label} + RL (15 stages)"
    else:
        labels = rl_keys
        hybrid_ratios = [rl_results[key]["hybrid_ratio"] * 100 for key in rl_keys]
        stage_note = "RL (15 stages)"
    
    fig, ax = plt.subplots(figsize=(18, 8))  # Increased figure size
    x = np.arange(len(labels))
    width = 0.6
    
    # Generate colors based on value intensity (hybrid only)
    # Function to get color based on value (0-100)
    def get_intensity_color(value, cmap_name='Oranges', min_val=0, max_val=100):
        cmap = plt.get_cmap(cmap_name)
        # Normalize value to 0.3-0.9 range to ensure visibility (not too light, not too dark)
        norm_val = 0.3 + 0.6 * (value - min_val) / (max_val - min_val + 1e-6)
        return cmap(norm_val)

    # Determine min/max for normalization across all data points
    all_hybrid = hybrid_ratios
    min_h, max_h = min(all_hybrid), max(all_hybrid)

    hybrid_colors = [get_intensity_color(v, 'Oranges', min_h, max_h) for v in hybrid_ratios]
    
    bars = ax.bar(
        x,
        hybrid_ratios,
        width,
        label="Hybrid Strategy",
        color=hybrid_colors,
        edgecolor='grey',
        linewidth=0.6,
        zorder=2
    )

    ax.plot(
        x,
        hybrid_ratios,
        color="#b22222",
        marker="o",
        linewidth=3,
        markersize=7,
        label="Hybrid Trend",
        zorder=4
    )
    
    # Increased font sizes
    ax.set_xlabel("Data Group", fontsize=21, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=21, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=19)
    ax.tick_params(axis='y', labelsize=19)
    ax.grid(axis="y", alpha=0.3)

    
    # Custom legend for color intensity explanation
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=plt.get_cmap('Oranges')(0.6), edgecolor='grey', label='Hybrid Strategy'),
        Line2D([0], [0], color='#b22222', lw=3, marker='o', label='Hybrid Trend'),
        Patch(facecolor='none', edgecolor='none', label='(Color depth ∝ Percentage)')
    ]
    fig.suptitle(
        f"{exp_label} | Single vs Hybrid Strategy Distribution ({stage_note})",
        fontsize=23,
        fontweight="bold",
        y=0.95
    )
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        fontsize=15,
        title="Legend"
    )
    legend.get_title().set_fontsize(16)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight='bold'
        )
    
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    output_path = os.path.join(output_dir, "single_vs_hybrid_ratio.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_pdf_path = os.path.join(output_dir, "single_vs_hybrid_ratio.pdf")
    plt.savefig(output_pdf_path, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    print(f"  Saved to: {output_pdf_path}")
    plt.close()


def plot_hybrid_evolution(
    sft_results: Optional[Dict],
    rl_results: Dict,
    output_dir: str,
    exp_label: str,
    baseline_label: str,
    include_sft: bool = True
):
    print("\nGenerating hybrid evolution plot...")
    
    rl_keys = sorted(rl_results.keys(), key=get_step_sort_key)
    if include_sft:
        if sft_results is None:
            raise ValueError("sft_results is required when include_sft is True.")
        labels = [baseline_label] + rl_keys
        hybrid_ratios = [sft_results["hybrid_ratio"] * 100]
        hybrid_ratios.extend([rl_results[key]["hybrid_ratio"] * 100 for key in rl_keys])
        stage_note = f"{baseline_label} + RL (15 stages)"
    else:
        labels = rl_keys
        hybrid_ratios = [rl_results[key]["hybrid_ratio"] * 100 for key in rl_keys]
        stage_note = "RL (15 stages)"
    
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(labels))
    
    ax.fill_between(x, hybrid_ratios, color="#f6b26b", alpha=0.35)
    line = ax.plot(x, hybrid_ratios, color="#c0392b", linewidth=3)
    ax.scatter(x, hybrid_ratios, s=70, color="#c0392b", edgecolor="white", linewidth=0.8, zorder=3)
    
    for xi, yi in zip(x, hybrid_ratios):
        ax.annotate(
            f"{yi:.1f}%",
            xy=(xi, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_xlabel("Data Group", fontsize=20, fontweight='bold')
    ax.set_ylabel("Hybrid Strategy (%)", fontsize=20, fontweight='bold')
    ax.set_title(
        f"{exp_label} | Hybrid Strategy Evolution ({stage_note})",
        fontsize=22,
        fontweight="bold",
        pad=18
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis="y", alpha=0.3)
    
    legend = ax.legend([line[0]], ["Hybrid Ratio"], loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=14, title="Trend")
    legend.get_title().set_fontsize(15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(output_dir, "hybrid_ratio_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_pdf_path = os.path.join(output_dir, "hybrid_ratio_evolution.pdf")
    plt.savefig(output_pdf_path, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    print(f"  Saved to: {output_pdf_path}")
    plt.close()



def plot_strategy_distribution(data_results: Dict, title: str, output_path: str, exp_label: str):
    """Plot individual strategy distribution as pie chart"""
    print(f"\nGenerating strategy distribution pie chart: {title}...")
    
    individual_dist = data_results['individual_distribution']
    if not individual_dist:
        print(f"  No data to plot for {title}")
        return
    
    # Sort by strategy ID
    sorted_items = sorted(individual_dist.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
    
    labels = [f"{STRATEGY_NAMES.get(k, k)} ({k})" for k, v in sorted_items]
    sizes = [v for k, v in sorted_items]
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 14})  # Increased from 9
    
    # Improve label visibility
    for text in texts:
        text.set_fontsize(14)  # Increased from 9
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)  # Increased from 8
    
    ax.set_title(f"{exp_label} | {title}", fontsize=22, fontweight='bold', pad=24)  # Increased from 14
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()


def plot_strategy_bar_comparison(
    sft_results: Optional[Dict],
    rl_results: Dict,
    output_dir: str,
    exp_label: str,
    baseline_label: str,
    include_sft: bool = True
):
    print("\nGenerating strategy distribution comparison bar chart...")

    # Ensure all 20 strategy categories (0-19) are always shown, even if count is 0.
    all_strategies = [str(i) for i in range(20)]
    
    groups = sorted(rl_results.keys(), key=get_step_sort_key)
    if include_sft:
        groups = [baseline_label] + groups
    
    data_matrix = []
    for group in groups:
        if group == baseline_label:
            if sft_results is None:
                raise ValueError("sft_results is required when include_sft is True.")
            total = sum(sft_results['individual_distribution'].values())
            percentages = [sft_results['individual_distribution'].get(s, 0) / total * 100 if total > 0 else 0 
                          for s in all_strategies]
        else:
            total = sum(rl_results[group]['individual_distribution'].values())
            percentages = [rl_results[group]['individual_distribution'].get(s, 0) / total * 100 if total > 0 else 0
                          for s in all_strategies]
        data_matrix.append(percentages)
    
    cmap_groups = cm.viridis
    n_groups = len(groups)
    group_colors = []
    for i, g in enumerate(groups):
        if g == baseline_label:
            group_colors.append("#444444")
        else:
            t = 0.2 + 0.7 * i / max(1, n_groups - 1)
            group_colors.append(cmap_groups(t))

    fig, axes = plt.subplots(2, 1, figsize=(18, 14))  # Increased figure size
    
    strategies_part1 = [s for s in all_strategies if int(s) < 10]
    data_part1 = [[row[all_strategies.index(s)] for s in strategies_part1] for row in data_matrix]
    
    x1 = np.arange(len(strategies_part1))
    width = 0.8 / len(groups)
    
    for i, group in enumerate(groups):
        offset = (i - len(groups)/2 + 0.5) * width
        axes[0].bar(x1 + offset, data_part1[i], width, label=group, color=group_colors[i])
    
    axes[0].set_xlabel('Strategy ID', fontsize=24, fontweight='bold')  # Increased
    axes[0].set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')  # Increased
    axes[0].set_title(
        f"{exp_label} | Strategy Distribution (Strategies 0-9)",
        fontsize=30,
        fontweight='bold'
    )  # Increased
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([f"{s}\n{STRATEGY_NAMES.get(s, s)[:15]}" for s in strategies_part1], fontsize=14)  # Increased from 8
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].set_ylim(0, 35)
    legend0 = axes[0].legend(loc='upper right', fontsize=16, title="Stage")
    legend0.get_title().set_fontsize(16)
    axes[0].grid(axis='y', alpha=0.3)
    
    strategies_part2 = [s for s in all_strategies if int(s) >= 10]
    data_part2 = [[row[all_strategies.index(s)] for s in strategies_part2] for row in data_matrix]
    
    x2 = np.arange(len(strategies_part2))
    
    for i, group in enumerate(groups):
        offset = (i - len(groups)/2 + 0.5) * width
        axes[1].bar(x2 + offset, data_part2[i], width, label=group, color=group_colors[i])
    
    axes[1].set_xlabel('Strategy ID', fontsize=24, fontweight='bold')  # Increased
    axes[1].set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')  # Increased
    axes[1].set_title(
        f"{exp_label} | Strategy Distribution (Strategies 10-19)",
        fontsize=30,
        fontweight='bold'
    )  # Increased
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([f"{s}\n{STRATEGY_NAMES.get(s, s)[:15]}" for s in strategies_part2], fontsize=14)  # Increased from 8
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].set_ylim(0, 10)
    legend1 = axes[1].legend(loc='upper right', fontsize=16, title="Stage")
    legend1.get_title().set_fontsize(16)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "strategy_distribution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_pdf_path = os.path.join(output_dir, "strategy_distribution_comparison.pdf")
    plt.savefig(output_pdf_path, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    print(f"  Saved to: {output_pdf_path}")
    plt.close()


def plot_diversity_evolution(rl_results: Dict, output_dir: str, exp_label: str):
    print("\nGenerating diversity evolution plot...")
    
    step_groups = sorted(rl_results.keys(), key=get_step_sort_key)
    hybrid_ratios = [rl_results[k]['hybrid_ratio'] * 100 for k in step_groups]
    
    strategy_varieties = []
    for k in step_groups:
        unique_strategies = len(rl_results[k]['individual_distribution'])
        strategy_varieties.append(unique_strategies)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))  # Increased figure size
    
    x = np.arange(len(step_groups))
    
    color1 = '#d62728'
    ax1.set_xlabel('Training Progress (Step Range)', fontsize=19, fontweight='bold')  # Increased
    ax1.set_ylabel('Hybrid Strategy Ratio (%)', color=color1, fontsize=19, fontweight='bold')  # Increased
    ax1.fill_between(x, 0, hybrid_ratios, color=color1, alpha=0.15)
    line1 = ax1.plot(x, hybrid_ratios, color=color1, marker='o', linewidth=3, markersize=10, label='Hybrid Ratio')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(step_groups, rotation=45, ha='right', fontsize=15)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.set_ylabel('Number of Unique Strategies', color=color2, fontsize=19, fontweight='bold')  # Increased
    line2 = ax2.plot(x, strategy_varieties, color=color2, marker='s', linewidth=3, markersize=10, label='Strategy Variety')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=15)
    
    plt.title(
        f"{exp_label} | Strategy Diversity Evolution During RL Training",
        fontsize=22,
        fontweight='bold',
        pad=24
    )  # Increased
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper left', fontsize=15, title="Metrics")  # Increased
    legend.get_title().set_fontsize(16)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "diversity_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()



def save_statistics(sft_results: Dict, rl_results: Dict, output_path: str):
    """Save detailed statistics to JSON"""
    print(f"\nSaving statistics to {output_path}...")
    
    results = {
        "sft": sft_results,
        "rl": rl_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved successfully")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Pattern Diversity: SFT vs RL")
    parser.add_argument(
        "--sft-dir",
        default=DEFAULT_SFT_DIR,
        help="Directory containing baseline classified files"
    )
    parser.add_argument(
        "--rl-dir",
        default="",
        help="Directory containing RL classified files"
    )
    parser.add_argument(
        "--experiment",
        choices=["attacker-base", "attacker-only", "attacker-sft"],
        default=None,
        help="Use preset RL directory for a specific experiment"
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save visualization results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PATTERN DIVERSITY ANALYSIS: BASELINE vs RL")
    print("=" * 80)
    
    exp_key = None
    if args.experiment:
        exp_key = args.experiment
        args.rl_dir = EXPERIMENT_RL_DIRS[exp_key]
    else:
        if "A-q257bi" in args.rl_dir:
            exp_key = "attacker-base"
        elif "q257bisft_wocode-only" in args.rl_dir:
            exp_key = "attacker-only"
        elif "A-sft_wocode" in args.rl_dir:
            exp_key = "attacker-sft"
        else:
            exp_key = "attacker-sft"
    
    exp_label = {
        "attacker-base": "Attacker-base",
        "attacker-only": "Attacker-only",
        "attacker-sft": "Attacker-SFT",
    }[exp_key]
    baseline_label = BASELINE_LABELS.get(exp_key, "SFT")
    baseline_dir = args.sft_dir
    if exp_key in BASELINE_DIRS and args.sft_dir == DEFAULT_SFT_DIR:
        baseline_dir = BASELINE_DIRS[exp_key]
    include_sft = True

    sft_results = analyze_baseline_data(baseline_dir, baseline_label)
    
    # Analyze RL data
    rl_results = analyze_rl_data(args.rl_dir)
    rl_results_single = analyze_rl_data_fifteen_stage(args.rl_dir)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Single vs Hybrid ratio comparison
    plot_single_vs_hybrid(
        sft_results,
        rl_results_single,
        args.output_dir,
        exp_label,
        baseline_label,
        include_sft
    )
    plot_hybrid_evolution(
        sft_results,
        rl_results_single,
        args.output_dir,
        exp_label,
        baseline_label,
        include_sft
    )
    
    # 2. Baseline strategy distribution pie chart
    if include_sft:
        plot_strategy_distribution(
            sft_results,
            f"{baseline_label} Strategy Distribution (Split by Individual Strategies)",
            os.path.join(args.output_dir, "sft_strategy_distribution.png"),
            exp_label
        )
    
    # 3. RL strategy distribution pie charts for each interval
    for group_key in sorted(rl_results.keys(), key=get_step_sort_key):
        plot_strategy_distribution(
            rl_results[group_key],
            f"RL Strategy Distribution: {group_key}",
            os.path.join(args.output_dir, f"rl_strategy_distribution_{group_key}.png"),
            exp_label
        )
    
    # 4. Strategy distribution comparison bar chart
    plot_strategy_bar_comparison(
        sft_results,
        rl_results,
        args.output_dir,
        exp_label,
        baseline_label,
        include_sft
    )
    
    # 5. Diversity evolution plot
    plot_diversity_evolution(rl_results, args.output_dir, exp_label)
    
    # Save statistics
    save_statistics(
        sft_results,
        rl_results,
        os.path.join(args.output_dir, "diversity_statistics.json")
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
