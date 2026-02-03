"""
Attacker Prompt Classification System
Supports coarse (5 main strategies) and fine-grained (20 subcategories) modes.
Fine-grained standards are defined in prompts_grained.py.
"""

import json
import logging
import os
import re
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
from uuid import uuid4
from openai import OpenAI
import argparse
from datetime import datetime

# Import prompts from prompts moduleå

from prompts_grained import get_system_prompt as get_system_prompt_grained
from prompts_grained import get_user_prompt as get_user_prompt_grained
from prompts_grained import FINE_GRAINED_CATEGORIES

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Classification Category Definitions
# ============================================================================

# NOTE (2025-01-16): Fine-grained (0-19) taxonomy is now the source of truth.
CLASSIFICATION_CATEGORIES = {
    "ISS": {"id": 1, "name": "Identity & Scenario Simulation"},
    "SAI": {"id": 2, "name": "Semantic Ambiguity & Induction"},
    "LRPA": {"id": 3, "name": "Logical Reversal & Premise Assumption"},
    "CC": {"id": 4, "name": "Complex Construction"},
    "SFM": {"id": 5, "name": "Structural & Format Manipulation"}
}

COMBINED_STRATEGIES = {
    "ISS_SAI": 6, "ISS_LRPA": 7, "ISS_CC": 8, "ISS_SFM": 9,
    "SAI_LRPA": 10, "SAI_CC": 11, "SAI_SFM": 12,
    "LRPA_CC": 13, "LRPA_SFM": 14, "CC_SFM": 15,
    "MULTIPLE": 16
}

FINE_GRAINED_ID_SET = (
    {entry["id"] for entry in FINE_GRAINED_CATEGORIES.values()}
    if FINE_GRAINED_CATEGORIES
    else set(range(20))
)


# ============================================================================
# JSON Processing
# ============================================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from API response text.
    Supports multiple formats:
    1. <answer>JSON</answer> tags (preferred format)
    2. ```json JSON ``` code blocks
    3. Direct JSON object
    4. Bare JSON in text
    """
    if not text:
        return None
    
    # Method 1: Extract from <answer></answer> tags (preferred for thinking models)
    import re
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
    
    # If multiple answer tags found (model repetition issue), try each one
    if answer_matches:
        for answer_content in answer_matches:
            answer_content = answer_content.strip()
            # Skip if content looks incomplete or is just a fragment
            if len(answer_content) < 10:
                continue
            try:
                # Try to parse as JSON
                parsed = json.loads(answer_content)
                # Return the first successfully parsed JSON
                return parsed
            except json.JSONDecodeError:
                # Try to fix common issues
                # Remove markdown code blocks if present
                if '```' in answer_content:
                    answer_content = re.sub(r'```json\s*|\s*```', '', answer_content, flags=re.IGNORECASE)
                    try:
                        return json.loads(answer_content.strip())
                    except json.JSONDecodeError:
                        continue
                # Continue to next match if this one fails
                continue
    
    # Method 2: Try extracting from ```json ``` code blocks
    json_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Method 3: Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Method 4: Try to find JSON object in text
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    
    return None


def validate_classification(data: Dict[str, Any], use_grained: bool = False) -> bool:
    """Validate classification response format"""
    required_fields = [
        "primary_strategy",
        "all_strategies",
        "strategy_details",
        "combined_strategy_code",
        "reasoning"
    ]
    
    for field in required_fields:
        if field not in data:
            return False

    if not isinstance(data.get("combined_strategy_code"), str):
        return False
    if not isinstance(data.get("reasoning"), str):
        return False
    
    if use_grained:
        # Fine-grained validation: primary_strategy can be int (0-19) or string ("0_3")
        primary = data.get("primary_strategy")
        if isinstance(primary, int):
            if primary not in FINE_GRAINED_ID_SET:
                return False
        elif isinstance(primary, str):
            # Could be single ID like "0" or combined like "0_3"
            try:
                ids = [int(x) for x in primary.split("_")]
                if any(id not in FINE_GRAINED_ID_SET for id in ids):
                    return False
            except (ValueError, AttributeError):
                return False
        else:
            return False
        
        # Validate all_strategies list contains integers 0-19
        if not isinstance(data.get("all_strategies"), list):
            return False
        
        for strategy in data.get("all_strategies", []):
            if not isinstance(strategy, int) or strategy not in FINE_GRAINED_ID_SET:
                return False
    else:
        # Coarse-grained validation (original logic)
        valid_strategies = set(CLASSIFICATION_CATEGORIES.keys()) | set(COMBINED_STRATEGIES.keys())
        
        if data.get("primary_strategy") not in valid_strategies:
            return False
        
        if not isinstance(data.get("all_strategies"), list):
            return False
        
        for strategy in data.get("all_strategies", []):
            if strategy not in valid_strategies:
                return False
    
    return True


# ============================================================================
# API Client
# ============================================================================

class ClassificationClient:
    """Classification API client for OpenAI-compatible endpoints"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "orm"):
        """Initialize API client with credentials"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Optional[str]:
        """Call API to classify a prompt"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None


# ============================================================================
# Processing Logic
# ============================================================================

def classify_single_record(
    client: ClassificationClient,
    question: str,
    attacker_answer: str,
    step: int,
    exchange_id: int,
    scores: Dict[str, float],
    use_grained: bool = False
) -> Dict[str, Any]:
    """Classify a single record"""
    record_id = str(uuid4())
    
    # Get prompts from prompts module
    if use_grained:
        if get_system_prompt_grained is None:
            raise ImportError("prompts_grained module not found. Cannot use fine-grained classification.")
        system_prompt = get_system_prompt_grained()
        user_prompt = get_user_prompt_grained(question, attacker_answer)
    else:
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(question, attacker_answer)
    
    # Call API
    raw_response = client.classify(system_prompt, user_prompt)
    
    if not raw_response:
        return {
            "record_id": record_id,
            "question": question,
            "attacker_answer": attacker_answer,
            "step": step,
            "exchange_id": exchange_id,
            "scores": scores,
            "status": "failed",
            "error": "API call failed"
        }
    
    # Parse JSON
    classification = extract_json_from_text(raw_response)
    
    if not classification or not validate_classification(classification, use_grained):
        return {
            "record_id": record_id,
            "question": question,
            "attacker_answer": attacker_answer,
            "step": step,
            "exchange_id": exchange_id,
            "scores": scores,
            "status": "failed",
            "error": "JSON parsing or validation failed",
            "raw_response": raw_response
        }
    
    # Add numeric IDs
    if use_grained:
        primary_value = classification.get("primary_strategy")
        primary_id = None
        if isinstance(primary_value, int):
            primary_id = primary_value
        elif isinstance(primary_value, str):
            parts = [p for p in primary_value.split("_") if p.isdigit()]
            if len(parts) == 1:
                primary_id = int(parts[0])

        all_ids = [
            s for s in classification.get("all_strategies", [])
            if isinstance(s, int)
        ]
        combined_id = classification.get("combined_strategy_code")
    else:
        primary_id = CLASSIFICATION_CATEGORIES.get(
            classification.get("primary_strategy", ""), {}
        ).get("id")
        
        all_ids = []
        for strategy in classification.get("all_strategies", []):
            if strategy in CLASSIFICATION_CATEGORIES:
                all_ids.append(CLASSIFICATION_CATEGORIES[strategy]["id"])
            elif strategy in COMBINED_STRATEGIES:
                all_ids.append(COMBINED_STRATEGIES[strategy])
        
        combined_id = primary_id if len(all_ids) <= 1 else (
            COMBINED_STRATEGIES.get("_".join(sorted(classification.get("all_strategies", [])))) or 16
        )
    
    classification.update({
        "primary_id": primary_id,
        "all_ids": all_ids,
        "combined_id": combined_id
    })
    
    return {
        "record_id": record_id,
        "question": question,
        "attacker_answer": attacker_answer,
        "step": step,
        "exchange_id": exchange_id,
        "scores": scores,
        "classification": classification,
        "status": "success",
        "raw_response": raw_response
    }


def process_sft_data(
    client: ClassificationClient,
    sft_data_dir: str,
    output_dir: str,
    use_grained: bool = False,
    start_file_index: int = 0
) -> Dict[str, Any]:
    """
    Process SFT training data from sorry_bench dataset
    
    Args:
        client: Classification API client
        sft_data_dir: Directory containing SFT JSONL files
        output_dir: Directory to save classification results
        use_grained: Whether to use fine-grained classification
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load base questions (original questions)
    base_file = os.path.join(sft_data_dir, "question.jsonl")
    if not os.path.exists(base_file):
        logger.error(f"Base question file not found: {base_file}")
        return {"status": "failed", "message": "Base question file not found"}
    
    # Read all base questions into a dict indexed by question_id
    base_questions = {}
    with open(base_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                question_id = record.get("question_id")
                turns = record.get("turns", [])
                if question_id and turns:
                    base_questions[question_id] = turns[0] if len(turns) > 0 else ""
    
    logger.info(f"Loaded {len(base_questions)} base questions")
    
    # Find all modified question files (excluding the base file)
    import glob
    pattern = os.path.join(sft_data_dir, "question_*.jsonl")
    files = sorted(glob.glob(pattern))
    skip_keywords = ("translate", "ascii", "atbash", "caesar", "morse")
    files = [f for f in files if not any(k in Path(f).name.lower() for k in skip_keywords)]
    
    if not files:
        logger.warning(f"No modified question files found in {sft_data_dir}")
        return {"status": "failed", "message": "No modified question files found"}
    
    logger.info(f"Found {len(files)} modified question files")
    
    # Apply start_file_index filter
    if start_file_index > 0:
        if start_file_index >= len(files):
            logger.error(f"start_file_index {start_file_index} exceeds total files {len(files)}")
            return {"status": "failed", "message": f"start_file_index {start_file_index} out of range"}
        files = files[start_file_index:]
        logger.info(f"Starting from file index {start_file_index}, processing {len(files)} files")
    
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Process each modified question file
    for file_idx, filepath in enumerate(files):
        filename = Path(filepath).name
        logger.info(f"Processing file {file_idx + 1}/{len(files)}: {filename}")
        
        # Extract prompt_style from filename (e.g., "question_ascii.jsonl" -> "ascii")
        prompt_style = filename.replace("question_", "").replace(".jsonl", "")
        
        # Read modified questions
        classified_records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                if line_number % 2 == 0:
                    continue
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    question_id = record.get("question_id")
                    category = record.get("category", "")
                    turns = record.get("turns", [])
                    file_prompt_style = record.get("prompt_style", prompt_style)
                    
                    if question_id not in base_questions:
                        logger.warning(f"Question ID {question_id} not found in base questions")
                        continue
                    
                    if not turns or len(turns) == 0:
                        logger.warning(f"No turns found for question ID {question_id}")
                        continue
                    
                    # Get original and modified questions
                    original_question = base_questions[question_id]
                    modified_question = turns[0]
                    
                    # Classify the modified question
                    result = classify_single_record(
                        client=client,
                        question=original_question,
                        attacker_answer=modified_question,
                        step=-1,  # SFT data doesn't have step
                        exchange_id=question_id,
                        scores={},
                        use_grained=use_grained
                    )
                    
                    # Add SFT-specific metadata
                    result["question_id"] = question_id
                    result["category"] = category
                    result["prompt_style"] = file_prompt_style
                    result["original_question"] = original_question
                    result["modified_question"] = modified_question
                    
                    classified_records.append(result)
                    
                    if result.get("status") == "success":
                        total_success += 1
                    else:
                        total_failed += 1
                    
                    total_processed += 1
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line in {filename}: {e}")
                    total_failed += 1
                    continue
                except Exception as e:
                    logger.error(f"Error processing record in {filename}: {e}")
                    total_failed += 1
                    continue
        
        # Save results for this file
        output_filename = f"classified_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in classified_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Saved results to: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
    
    # Generate summary
    summary = {
        "total_files_processed": len(files),
        "total_records_processed": total_processed,
        "successful_records": total_success,
        "failed_records": total_failed,
        "success_rate": f"{total_success / max(1, total_processed) * 100:.2f}%",
        "output_directory": output_dir,
        "mode": "sft",
        "use_grained_classification": use_grained
    }
    
    summary_path = os.path.join(output_dir, "sft_classification_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("SFT Data Processing completed!")
    logger.info("=" * 80)
    logger.info(f"Total records: {total_processed}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success rate: {summary['success_rate']}")
    logger.info(f"Fine-grained classification: {use_grained}")
    logger.info("=" * 80)
    
    return {"status": "completed", "summary": summary}


def process_extracted_files(
    client: ClassificationClient,
    extracted_dir: str,
    output_dir: str,
    max_files: Optional[int] = None,
    specific_steps: Optional[List[int]] = None,
    use_grained: bool = False,
    process_all_lines: bool = False
) -> Dict[str, Any]:
    """Process all extracted JSONL files"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # List files to process
    if specific_steps:
        files = [
            os.path.join(extracted_dir, f"extracted_step_{step}.jsonl")
            for step in specific_steps
        ]
        files = [f for f in files if os.path.exists(f)]
    else:
        import glob
        files = sorted(glob.glob(os.path.join(extracted_dir, "extracted_step_*.jsonl")))
        if max_files:
            files = files[:max_files]
    
    if not files:
        logger.warning("No extracted files found")
        return {"status": "failed", "message": "No extracted files found"}
    
    logger.info(f"Found {len(files)} files to process")
    
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Process each file
    for file_idx, filepath in enumerate(files):
        filename = Path(filepath).name
        logger.info(f"Processing file {file_idx + 1}/{len(files)}: {filename}")
        
        # Read input file
        records = []
        line_number = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    if line.strip():
                        # Process all lines or only lines where line_number % 4 == 1
                        if process_all_lines or (line_number % 4 == 1):
                            records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            continue
        
        # Classify each record
        classified_records = []
        for record_idx, record in enumerate(records):
            try:
                question = record.get("question", "")
                step = record.get("step")
                exchange_id = record.get("exchange_id")
                scores = record.get("scores", {})
                
                attacker = record.get("attacker", {})
                attacker_answer = attacker.get("parsed_answer", "")
                
                if not all([question, attacker_answer, step is not None, exchange_id is not None]):
                    logger.warning(f"Record {record_idx} missing required fields")
                    continue
                
                result = classify_single_record(
                    client, question, attacker_answer, step, exchange_id, scores, use_grained
                )
                
                classified_records.append(result)
                
                if result.get("status") == "success":
                    total_success += 1
                else:
                    total_failed += 1
                
                total_processed += 1
                
                if (record_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {record_idx + 1}/{len(records)} records")
            
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                total_failed += 1
                continue
        
        # Save results
        output_filename = f"classified_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in classified_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Saved results to: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
    
    # Generate summary
    summary = {
        "total_files_processed": len(files),
        "total_records_processed": total_processed,
        "successful_records": total_success,
        "failed_records": total_failed,
        "success_rate": f"{total_success / max(1, total_processed) * 100:.2f}%",
        "output_directory": output_dir
    }
    
    summary_path = os.path.join(output_dir, "classification_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("Processing completed!")
    logger.info("=" * 80)
    logger.info(f"Total records: {total_processed}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Success rate: {summary['success_rate']}")
    logger.info("=" * 80)
    
    return {"status": "completed", "summary": summary}


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point for the classification system"""
    parser = argparse.ArgumentParser(description="Attacker Prompt Classification System")
    
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--api-url",
        default="",
        help="API base URL"
    )
    
    parser.add_argument(
        "--model",
        default="",
        help="Model name to use"
    )
    
    parser.add_argument(
        "--extracted-dir",
        default="",
        help="Directory containing extracted JSONL files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save classification results"
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific step IDs to process"
    )
    
    parser.add_argument(
        "--mode",
        choices=["rl", "sft"],
        default="rl",
        help="Classification mode: 'rl' for RL training data, 'sft' for SFT training data"
    )
    
    parser.add_argument(
        "--sft-data-dir",
        default="",
        help="Directory containing SFT training data (for --mode sft)"
    )
    
    parser.add_argument(
        "--use-grained",
        action="store_true",
        help="Use fine-grained classification (20 subcategories 0-19)"
    )
    
    parser.add_argument(
        "--start-file-index",
        type=int,
        default=0,
        help="Start processing from this file index (0-based, e.g., 6 for 7th file)"
    )
    
    parser.add_argument(
        "--process-all-lines",
        action="store_true",
        help="Process all lines in each file (default: only process lines where line_number %% 4 == 1)"
    )
    
    parser.add_argument(
        "--step-start",
        type=int,
        default=None,
        help="Start step number for filtering files (e.g., 20)"
    )
    
    parser.add_argument(
        "--step-end",
        type=int,
        default=None,
        help="End step number for filtering files (e.g., 500)"
    )
    
    parser.add_argument(
        "--step-interval",
        type=int,
        default=None,
        help="Step interval for filtering files (e.g., 20)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Attacker Prompt Classification System")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Fine-grained classification: {args.use_grained}")
    logger.info(f"API URL: {args.api_url}")
    if args.mode == "rl":
        logger.info(f"Extracted directory: {args.extracted_dir}")
        logger.info(f"Process all lines: {args.process_all_lines}")
        if args.step_start is not None or args.step_end is not None:
            logger.info(f"Step range: {args.step_start or 'start'} to {args.step_end or 'end'}")
        if args.step_interval is not None:
            logger.info(f"Step interval: {args.step_interval}")
    else:
        logger.info(f"SFT data directory: {args.sft_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Initialize API client
    client = ClassificationClient(
        api_key=args.api_key,
        base_url=args.api_url,
        model=args.model
    )
    
    # Run processing based on mode
    if args.mode == "sft":
        result = process_sft_data(
            client,
            args.sft_data_dir,
            args.output_dir,
            use_grained=args.use_grained,
            start_file_index=args.start_file_index
        )
    else:  # rl mode
        # Generate specific_steps based on step range if provided
        specific_steps = args.steps
        if specific_steps is None and args.step_start is not None and args.step_end is not None and args.step_interval is not None:
            specific_steps = list(range(args.step_start, args.step_end + 1, args.step_interval))
            logger.info(f"Generated step list: {specific_steps}")
        
        result = process_extracted_files(
            client,
            args.extracted_dir,
            args.output_dir,
            max_files=args.max_files,
            specific_steps=specific_steps,
            use_grained=args.use_grained,
            process_all_lines=args.process_all_lines
        )
    
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
