import re

def extract_final_verdict(text):
    """
    Extracts the final verdict character ('A' or 'B') from the evaluation text.
    
    Args:
        text (str): The input text containing the evaluation and the final verdict.
    
    Returns:
        str: The extracted final verdict character ('A' or 'B'), or None if not found.
    """
    # Define the pattern to match the final verdict format
    pattern = r'\[\[([A-B])\]\]'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        # Return the extracted verdict character (either 'A' or 'B')
        return match.group(1)  # Extracts just the 'A' or 'B' character
    else:
        # Return None if no verdict found
        return None

def compute_score(model_output: str, groundtruth: str) -> bool:
    model_answer = extract_final_verdict(model_output)
    if model_answer == groundtruth:
        return 1.0
    else:
        return 0.0