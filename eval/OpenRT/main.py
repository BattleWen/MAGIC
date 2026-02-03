"""
Main entry point for running OpenRT experiments from YAML configuration files.
"""

import argparse
import logging
import os
import sys
import inspect
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# import OpenRT.judges

from OpenRT.core.orchestrator import Orchestrator
from OpenRT.core.registry import (
    attack_registry, model_registry, dataset_registry, 
    evaluator_registry, judge_registry
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate YAML configuration files."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load YAML config and substitute environment variables."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = ConfigLoader._substitute_env_vars(config)
            ConfigLoader._validate_config(config)
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
    
    @staticmethod
    def _substitute_env_vars(obj: Any) -> Any:
        """Recursively substitute ${VAR_NAME} with environment variables."""
        if isinstance(obj, dict):
            return {k: ConfigLoader._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigLoader._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            value = os.getenv(var_name)
            if value is None:
                logger.warning(f"Environment variable {var_name} not found")
            return value
        return obj
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate that required config sections exist."""
        required_sections = ['models', 'dataset', 'attack', 'evaluation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        if 'target_model' not in config['models']:
             raise ValueError("Config 'models' section must contain 'target_model'")


class ComponentFactory:
    """Factory for creating components from configuration using registries."""
    
    @staticmethod
    def _inject_args(cls, args: Dict[str, Any], **dependencies) -> Any:
        """
        Instantiate class injecting dependencies if they match the signature.
        Handles cases where different classes need different subsets of available objects.
        """
        sig = inspect.signature(cls.__init__)
        init_kwargs = args.copy()
        
        # Check if class accepts **kwargs
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        
        # Inject dependencies if they are in signature or if **kwargs is present
        for name, obj in dependencies.items():
            if name in sig.parameters or has_var_keyword:
                init_kwargs[name] = obj
        
        # If no **kwargs, filter out arguments that are not in signature to avoid TypeError
        if not has_var_keyword:
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in sig.parameters}
            
        return cls(**init_kwargs)

    @staticmethod
    def create_models(models_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create all models defined in the configuration."""
        models = {}
        for key, config in models_config.items():
            model_type = config.get('name')
            args = config.get('args', {})
            cls = model_registry.get(model_type)
            models[key] = cls(**args)
        
        # Ensure 'model' alias exists for 'target_model' as many components expect 'model'
        if 'target_model' in models:
            models['model'] = models['target_model']
            
        return models
    
    @staticmethod
    def create_dataset(config: Dict[str, Any]):
        """Create a dataset instance from config."""
        dataset_type = config.get('name')
        args = config.get('args', {})
        cls = dataset_registry.get(dataset_type)
        return cls(**args)
    
    @staticmethod
    def create_judge(config: Dict[str, Any], models: Dict[str, Any]):
        """Create a judge instance."""
        judge_type = config.get('name')
        args = config.get('args', {}).copy()
        cls = judge_registry.get(judge_type)
        
        # Strategy to select the LLM for the judge
        # 1. Look for explicit 'judge_model' in models
        # 2. Fallback to 'attacker_model'
        # 3. Fallback to 'target_model' (aliased as 'model')
        judge_model = models.get('judge_model') or models.get('attacker_model') or models.get('model')
        
        return ComponentFactory._inject_args(cls, args, judge_model=judge_model)
    
    @staticmethod
    def create_attack(config: Dict[str, Any], models: Dict[str, Any]):
        """Create an attack instance, handling internal judge if configured."""
        attack_type = config.get('name')
        args = config.get('args', {}).copy()
        
        # Handle internal judge if specified in attack args
        internal_judge = None
        if 'judge' in args:
            judge_config = args.pop('judge')
            # Create the internal judge using available models
            internal_judge = ComponentFactory.create_judge(judge_config, models)
        
        cls = attack_registry.get(attack_type)
        
        # Prepare dependencies: all models + the internal judge
        dependencies = models.copy()
        if internal_judge:
            dependencies['judge'] = internal_judge
            
        return ComponentFactory._inject_args(cls, args, **dependencies)
    
    @staticmethod
    def create_evaluator(config: Dict[str, Any], models: Dict[str, Any]):
        """Create an evaluator and its external judge."""
        # Config structure: { evaluator: {name:..., args:...}, judge: {name:..., args:...} }
        
        # 1. Create External Judge
        judge_config = config.get('judge')
        if not judge_config:
             raise ValueError("Evaluation section must contain a 'judge' configuration.")
        
        external_judge = ComponentFactory.create_judge(judge_config, models)
        
        # 2. Create Evaluator
        evaluator_config = config.get('evaluator')
        if not evaluator_config:
             raise ValueError("Evaluation section must contain an 'evaluator' configuration.")
             
        evaluator_type = evaluator_config.get('name')
        evaluator_args = evaluator_config.get('args', {})
        cls = evaluator_registry.get(evaluator_type)
        
        return ComponentFactory._inject_args(cls, evaluator_args, judge=external_judge)


class ExperimentRunner:
    """Run experiments from configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ConfigLoader.load(config_path)
        logger.info(f"Loaded config from: {config_path}")
    
    def run(self) -> None:
        """Execute the experiment."""
        try:
            # 1. Create Models
            logger.info("Creating models...")
            models = ComponentFactory.create_models(self.config['models'])
            
            # 2. Create Dataset
            dataset = ComponentFactory.create_dataset(self.config['dataset'])
            
            # 3. Create Attack (handles internal judge creation)
            logger.info(f"Creating attack: {self.config['attack']['name']}")
            attack = ComponentFactory.create_attack(self.config['attack'], models)
            
            # 4. Create Evaluator (handles external judge creation)
            logger.info("Creating evaluator...")
            evaluator = ComponentFactory.create_evaluator(self.config['evaluation'], models)
            
            # 5. Run Orchestrator
            logger.info("Starting experiment...")
            orchestrator = Orchestrator(
                model=models['model'], # target_model
                dataset=dataset,
                attack=attack,
                evaluator=evaluator
            )
            metrics, results = orchestrator.run()
            
            # Display results
            self._display_results(metrics, results)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
    
    def _display_results(self, metrics, results) -> None:
        """Display experiment results."""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT RESULTS")
        logger.info("="*60)
        logger.info(f"Attack Success Rate: {metrics.attack_success_rate:.2%}")
        logger.info(f"Total Samples: {len(results)}")
        
        successful_attacks = [r for r in results if r.success]
        if successful_attacks:
            logger.info(f"\nSuccessful Attacks: {len(successful_attacks)}")
            for i, result in enumerate(successful_attacks[:3], 1):
                logger.info(f"\n--- Example {i} ---")
                logger.info(f"Objective: {result.target[:100]}...")
                logger.info(f"Final Prompt: {result.final_prompt[:200]}...")
                logger.info(f"Response: {result.output_text[:200]}...")
        
        logger.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OpenRT experiments from YAML configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        runner = ExperimentRunner(args.config)
        runner.run()
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
