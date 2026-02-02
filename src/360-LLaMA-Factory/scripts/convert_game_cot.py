#!/usr/bin/env python3
"""Convert game_cot.jsonl records to the game.json format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSONL data that contains CoT outputs into "
            "the JSON array format used by game.json."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "game_cot.jsonl",
        help="Path to the jsonl file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "game.json",
        help="Destination path for the json file (default: %(default)s)",
    )
    return parser.parse_args()


def convert_records(input_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse line {line_no} of {input_path}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"Line {line_no} does not contain a JSON object.")
            records.append(sample)
    return records


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    converted = convert_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        json.dump(converted, target, ensure_ascii=False, indent=2)
        target.write("\n")

    print(f"Wrote {len(converted)} records to {output_path}")


if __name__ == "__main__":
    main()
