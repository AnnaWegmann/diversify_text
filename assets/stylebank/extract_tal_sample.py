#!/usr/bin/env python3
"""Extract utterance samples from TAL transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract utterances from data/tal/valid-transcripts-aligned.json and output JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/tal/valid-transcripts-aligned.json"),
        help="Input TAL JSON file (default: data/tal/valid-transcripts-aligned.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/tal/tal_output.json"),
        help="Output JSON file (default: data/tal/tal_output.json)",
    )
    parser.add_argument(
        "-m",
        "--max-sents",
        type=int,
        default=64,
        help="Maximum utterances to keep (default: 64)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum tokens per utterance (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per utterance (default: 100)",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def extract_utterances(
    input_path: Path,
    max_sents: int,
    min_tokens: int,
    max_tokens: int,
) -> list[str]:
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    utterances: list[str] = []
    for _episode, rows in data.items():
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            text = row.get("utterance")
            if not isinstance(text, str):
                continue

            text = normalize(text)
            if not text:
                continue
            token_count = len(text.split())
            if token_count < min_tokens or token_count > max_tokens:
                continue

            utterances.append(text)
            if len(utterances) >= max_sents:
                return utterances

    return utterances


def main() -> None:
    args = parse_args()
    utterances = extract_utterances(
        input_path=args.input,
        max_sents=args.max_sents,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    output = {"spoken_communication": utterances}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"spoken_communication": len(utterances)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()