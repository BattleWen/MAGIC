import unittest
import torch
from unittest.mock import patch, MagicMock

from src.classifier_models.llama_guard import LlamaGuard3
from src.classifier_models.base import SafetyClassifierOutput, PromptHarmfulness, ResponseHarmfulness


class TestLlamaGuard3(unittest.TestCase):
    """Test cases for the LlamaGuard3 class."""

    @patch('src.classifier_models.llama_guard.AutoModelForCausalLM')
    @patch('src.classifier_models.llama_guard.AutoTokenizer')
    def setUp(self, mock_tokenizer, mock_model):
        """Set up the test environment with mocked dependencies."""
        # Configure the mock tokenizer
        mock_tokenizer_instance = MagicMock()
        # Return valid indices that are within the tensor dimensions
        mock_tokenizer_instance.encode.return_value = [0, 1, 2, 3, 4]
        mock_tokenizer_instance.apply_chat_template.return_value = "formatted conversation"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Configure the mock model
        mock_model_instance = MagicMock()
        # Create a tensor with appropriate dimensions for the test
        mock_model_instance.return_value.logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]]).cpu()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize the LlamaGuard3 instance
        self.classifier = LlamaGuard3(batch_size=2)
        
        # Store the mocks for later use
        self.mock_tokenizer = mock_tokenizer_instance
        self.mock_model = mock_model_instance
        
        # Override CUDA tensors to use CPU for testing
        # Use indices that are within the tensor dimensions
        self.classifier.safe_loc = torch.tensor([0]).unsqueeze(-1).cpu()
        self.classifier.unsafe_loc = torch.tensor([1]).unsqueeze(-1).cpu()

    def test_initialization(self):
        """Test that the classifier initializes correctly."""
        self.assertEqual(self.classifier.model_name, "meta-llama/Llama-Guard-3-8B")
        self.assertEqual(self.classifier.batch_size, 2)
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.tokenizer)

    def test_required_input_fields(self):
        """Test that the required input fields are correct."""
        required_fields = self.classifier.get_required_input_fields()
        self.assertEqual(required_fields, ["prompt"])

    def test_optional_input_fields(self):
        """Test that the optional input fields are correct."""
        optional_fields = self.classifier.get_optional_input_fields()
        self.assertEqual(optional_fields, ["response"])

    def test_output_fields(self):
        """Test that the output fields are correct."""
        output_fields = self.classifier.get_output_fields()
        self.assertEqual(output_fields, ["prompt_harmfulness", "response_harmfulness"])

    @patch('torch.inference_mode')
    def test_classify_batch_prompt_only(self, mock_inference_mode):
        """Test classifying a batch with only prompts (no responses)."""

        # Create test inputs
        test_items = [
            {"prompt": "Hello, how are you?"},
            {"prompt": "What's the weather like today?"}
        ]
        
        # Call the classify method
        results = self.classifier.classify(test_items)
        print(results)
        
        # Verify the results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, SafetyClassifierOutput)
            self.assertIsNotNone(result.prompt_harmfulness)
            self.assertIsNone(result.response_harmfulness)
            self.assertIn("prompt_safe_prob", result.metadata)
            self.assertIn("prompt_unsafe_prob", result.metadata)

    @patch('torch.inference_mode')
    def test_classify_batch_with_responses(self, mock_inference_mode):
        """Test classifying a batch with both prompts and responses."""
        
        # Create test inputs
        test_items = [
            {"prompt": "Hello, how are you?", "response": "I'm doing well, thank you!"},
            {"prompt": "What's the weather like today?", "response": "It's sunny outside."}
        ]
        
        # Call the classify method
        results = self.classifier.classify(test_items)
        
        # Verify the results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, SafetyClassifierOutput)
            self.assertIsNotNone(result.prompt_harmfulness)
            self.assertIsNotNone(result.response_harmfulness)
            self.assertIn("prompt_safe_prob", result.metadata)
            self.assertIn("prompt_unsafe_prob", result.metadata)
            self.assertIn("response_safe_prob", result.metadata)
            self.assertIn("response_unsafe_prob", result.metadata)


if __name__ == '__main__':
    unittest.main() 