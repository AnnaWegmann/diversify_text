#!/usr/bin/env python3
"""Extract Chaucer sentence samples from the train.me file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Chaucer samples from data/chaucer/train.me and output JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/chaucer/train.me"),
        help="Input text file (default: data/chaucer/train.me)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/chaucer/chaucer_output.json"),
        help="Output JSON file (default: data/chaucer/chaucer_output.json)",
    )
    parser.add_argument(
        "-m",
        "--max-sents",
        type=int,
        default=64,
        help="Maximum number of sentences to keep (default: 64)",
    )
    parser.add_argument(
        "-w",
        "--min-words",
        type=int,
        default=8,
        help="Minimum words per sentence (default: 8)",
    )
    return parser.parse_args()


def normalize_sentence(line: str) -> str:
    sentence = " ".join(line.strip().split())
    if sentence.endswith(".") or sentence.endswith("!") or sentence.endswith("?") or sentence.endswith(";"):
        return sentence
    return ""


def extract_samples(input_path: Path, max_sents: int, min_words: int) -> list[str]:
    samples: list[str] = []

    with input_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            sentence = normalize_sentence(raw_line)
            if not sentence:
                continue
            if len(sentence.split()) < min_words:
                continue

            samples.append(sentence)
            if len(samples) >= max_sents:
                break

    return samples


def main() -> None:
    args = parse_args()
    sentences = extract_samples(
        input_path=args.input,
        max_sents=args.max_sents,
        min_words=args.min_words,
    )

    output = {"middle_english": sentences}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"middle_english": len(sentences)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()