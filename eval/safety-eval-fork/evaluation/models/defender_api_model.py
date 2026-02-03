"""
Defender API Model for StrongREJECT Alpacaeval Evaluation
extract answer from tag <think>...</think><answer>...</answer>
"""
import os
import re
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from evaluation.schemas import GeneratorModelBase


def extract_answer(text):
    """Extract content from <answer> tags, fallback to removing <think> tags."""
    if not text:
        return ""
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: remove <think> tags
    fallback = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return fallback.strip()


class DefenderAPIModel(GeneratorModelBase):
    """
    Defender API Model that calls deployed defender endpoint
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        model_input_template_path_or_name: str,
        batch_size: int | None = None,
        filter_classifier_name: str | None = None,
        filter_model_override_path: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        # Defender API configuration
        self.base_url = base_url or os.getenv(
            "DEFENDER_API_BASE_URL",
        )
        self.api_key = api_key or os.getenv("DEFENDER_API_KEY", "FAKE_API_KEY")
        self.model = model or os.getenv("DEFENDER_API_MODEL", "orm")
        
        # For defender_api, we use a simple instruction template
        # since the API endpoint should handle the chat template internally
        if model_input_template_path_or_name == "hf" or model_input_template_path_or_name is None:
            # Use simple instruction template for API calls
            self.model_input_template = "{instruction}"
            print(f"[DefenderAPI] Using simple instruction template for API calls")
        else:
            # If a custom template is provided, use it
            if model_input_template_path_or_name.endswith(".txt"):
                with open(model_input_template_path_or_name, "r") as f:
                    self.model_input_template = f.read()
            else:
                # Try to get template from templates module
                try:
                    from src.templates.single_turn import get_template
                    template_dict = get_template(
                        model_name_or_path=model_name_or_path,
                        chat_template=model_input_template_path_or_name,
                    )
                    self.model_input_template = template_dict["prompt"]
                except Exception as e:
                    print(f"[DefenderAPI] Warning: Failed to load template, using simple instruction: {e}")
                    self.model_input_template = "{instruction}"
        
        # Set required attributes for GeneratorModelBase
        self.model_name_or_path = model_name_or_path
        self.model_input_template_path_or_name = model_input_template_path_or_name
        self.batch_size = batch_size or 1
        self.filter_classifier_name = filter_classifier_name
        self.filter_model_override_path = filter_model_override_path
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        print(f"\n{'='*60}")
        print("Defender API Model Configuration")
        print(f"Base URL: {self.base_url}")
        print(f"Model: {self.model}")
        print(f"API Key: {self.api_key[:10]}..." if len(self.api_key) > 10 else f"API Key: {self.api_key}")
        print(f"Template: {self.model_input_template[:100]}...")
        print(f"{'='*60}\n")
    
    def load_model(self, model_name_or_path: str):
        """API model doesn't need to load model locally"""
        return None, None
    
    def generate_completions(
        self,
        inputs: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float | None = None,
        n: int = 1,
        stop: Optional[str | list[str]] = None,
        return_full_outputs: bool = False,
    ) -> list[str]:
        """
        Generate completions using Defender API
        
        Args:
            inputs: List of input dictionaries with 'instruction' field
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated completions (raw responses with <think><answer> tags)
        """
        formatted_inputs = [self.model_input_template.format(**d) for d in inputs]
        
        print(f"\n{'='*60}")
        print(f"Generating {len(formatted_inputs)} completions via Defender API")
        print(f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_new_tokens}")
        print(f"{'='*60}\n")
        
        outputs = []
        for i, prompt in enumerate(tqdm(formatted_inputs, desc="Defender API generation")):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                response = completion.choices[0].message.content
                outputs.append(response)
                
                # Print first response as sample
                if i == 0:
                    print(f"\n{'='*60}")
                    print("Sample Defender Response (first):")
                    print(f"Prompt: {prompt[:100]}...")
                    print(f"Response: {response[:200]}...")
                    print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"Error calling Defender API for input {i}: {e}")
                # Return empty response on error
                outputs.append("")
        
        print(f"\n{'='*60}")
        print(f"Generated {len(outputs)} completions from Defender API")
        print(f"{'='*60}\n")
        
        # Note: We return RAW responses with <think><answer> tags
        # The extract_answer() function will be called by the task evaluator
        return outputs
