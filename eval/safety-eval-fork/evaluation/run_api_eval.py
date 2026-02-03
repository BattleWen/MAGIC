#!/usr/bin/env python
"""
Standalone script for API-based Attacker-Defender evaluation.
This bypasses the eval.py framework since we don't need to load models.
"""
import sys
import os
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evaluation'))
sys.path.insert(0, os.path.dirname(__file__))

from evaluation.tasks.generation.harmbench import HarmbenchAttackerDefenderAPI

def main():
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f""
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 API-based Attacker-Defender Evaluation")
    print("="*70)
    print(f"📂 Results directory: {results_dir}\n")
    
    # Initialize API-based task
    print("📡 Initializing API clients...")
    task = HarmbenchAttackerDefenderAPI(
        classifier_model_name='Qwen3GuardAPI',
        attacker_api_url='',
        defender_api_url='',
        attacker_api_key='FAKE_API_KEY',
        defender_api_key='FAKE_API_KEY'
    )
    
    # Load data
    print("\n📥 Loading HarmBench dataset...")
    task.load_data()
    
    # Run evaluation
    print("\n🔥 Starting evaluation pipeline...")
    print("   1️⃣  Calling Attacker API to revise prompts")
    print("   2️⃣  Calling Defender API to generate responses")
    print("   3️⃣  Classifying with Qwen3Guard API\n")
    
    metrics, individual_results = task._evaluate(model=None)
    
    # Save results
    metrics_path = os.path.join(results_dir, 'metrics.json')
    results_path = os.path.join(results_dir, 'all_results.json')
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(individual_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ Evaluation completed successfully!")
    print("="*70)
    print(f"\n📊 Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
        else:
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print(f"\n📁 Results saved to:")
    print(f"   📄 {metrics_path}")
    print(f"   📄 {results_path}")
    print()

if __name__ == "__main__":
    main()
