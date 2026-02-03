import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command, cwd=None):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def get_latest_dir(path):
    if not os.path.exists(path):
        return None
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not dirs:
        return None
    # Sort by name (which is timestamp YYYY-MM-DD_HH-MM-SS)
    dirs.sort()
    return dirs[-1]

def stage_generate(x_teaming_dir):
    print("\n=== Step 1: Generating Attack Plans ===")
    run_command("python generate_attack_plans.py --config config/integration_config.yaml", cwd=x_teaming_dir)

def stage_attack(x_teaming_dir):
    print("\n=== Step 2: Executing Attacks ===")
    
    strategies_dir = x_teaming_dir / "strategies"
    latest_strategy_timestamp = get_latest_dir(strategies_dir)
    
    if not latest_strategy_timestamp:
        print("Error: No strategies generated. Please run 'generate' stage first.")
        sys.exit(1)
        
    print(f"Using latest strategies from: {latest_strategy_timestamp}")
    
    source_plan = strategies_dir / latest_strategy_timestamp / "attack_plans.json"
    dest_plan = strategies_dir / "attack_plans.json"
    
    if not source_plan.exists():
        print(f"Error: attack_plans.json not found in {source_plan.parent}")
        sys.exit(1)

    print(f"Copying {source_plan} to {dest_plan}")
    shutil.copy2(source_plan, dest_plan)
    
    run_command("python main.py --config config/integration_config.yaml", cwd=x_teaming_dir)

def stage_analyze(x_teaming_dir):
    print("\n=== Step 3: Viewing Results ===")
    
    attacks_dir = x_teaming_dir / "attacks"
    latest_attack_timestamp = get_latest_dir(attacks_dir)
    
    if not latest_attack_timestamp:
        print("Error: No attacks executed. Please run 'attack' stage first.")
        sys.exit(1)
        
    print(f"Analyzing latest attack results: {latest_attack_timestamp}")
    run_command(f"python analytics/metrics.py {latest_attack_timestamp}", cwd=x_teaming_dir)

def main():
    parser = argparse.ArgumentParser(description="Run X-Teaming Benchmark Stages")
    parser.add_argument("--stage", choices=["generate", "attack", "analyze", "all"], default="all",
                        help="Select the stage to run: generate (plans), attack (execute), analyze (metrics), or all (default)")
    args = parser.parse_args()

    # Define paths
    base_dir = Path(__file__).parent.absolute()
    x_teaming_dir = base_dir / "evaluation" / "tasks" / "generation" / "x-teaming"
    
    if not x_teaming_dir.exists():
        print(f"Error: X-Teaming directory not found at {x_teaming_dir}")
        sys.exit(1)

    print(f"Working directory: {x_teaming_dir}")

    if args.stage == "all":
        stage_generate(x_teaming_dir)
        stage_attack(x_teaming_dir)
        stage_analyze(x_teaming_dir)
    elif args.stage == "generate":
        stage_generate(x_teaming_dir)
    elif args.stage == "attack":
        stage_attack(x_teaming_dir)
    elif args.stage == "analyze":
        stage_analyze(x_teaming_dir)

if __name__ == "__main__":
    main()
