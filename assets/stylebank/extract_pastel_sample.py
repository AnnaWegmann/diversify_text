#!/usr/bin/env python3
"""Extract Pastel sentence samples grouped by persona dimensions.

Rules:
- Read JSON files from data/pastel/test.
- For each file, combine output.sentences into one single sentence string.
- Ensure the combined sentence ends with a full stop.
- Group sentences by these persona fields: politics, age, education, ethnic, gender.
- Output JSON keys as <field>_<value> with values as sentence lists.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PERSONA_FIELDS = ["politics", "age", "education", "ethnic", "gender"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Pastel test samples grouped by persona values and output JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/pastel/test"),
        help="Input directory containing Pastel JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/pastel/pastel_output.json"),
        help="Output JSON file path (default: data/pastel/pastel_output.json)",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=64,
        help="Maximum number of sentences per persona group (default: 64)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum words per merged sentence (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(text.split())


def starts_with_all_caps(text: str, window: int = 40, min_letters: int = 8) -> bool:
    prefix = text.lstrip()[:window]
    letters = [ch for ch in prefix if ch.isalpha()]
    if len(letters) < min_letters:
        return False
    return all(ch.isupper() for ch in letters)


def to_single_sentence(sentences: list[str]) -> str:
    merged = " ".join(s.strip() for s in sentences if isinstance(s, str) and s.strip())
    merged = " ".join(merged.split())
    if not merged:
        return ""
    if not merged.endswith("."):
        merged += "."
    return merged


def extract_samples(
    input_dir: Path,
    max_per_group: int,
    min_words: int,
    seed: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}

    for json_path in sorted(input_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as fh:
            record = json.load(fh)

        persona = record.get("persona") or {}
        output_sentences = record.get("output.sentences") or []

        sentence = to_single_sentence(output_sentences)
        if not sentence:
            continue
        if starts_with_all_caps(sentence):
            continue
        if word_count(sentence) < min_words:
            continue

        for field in PERSONA_FIELDS:
            value = persona.get(field)
            if not value:
                continue
            key = f"{field}_{value}"
            grouped.setdefault(key, []).append(sentence)

    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for key, candidates in grouped.items():
        if len(candidates) > max_per_group:
            candidates = rng.sample(candidates, max_per_group)
        sampled[key] = candidates

    sampled = {k: v for k, v in sampled.items() if len(v) >= 10}
    return dict(sorted(sampled.items()))


def main() -> None:
    args = parse_args()
    data = extract_samples(
        input_dir=args.input_dir,
        max_per_group=args.max_per_group,
        min_words=args.min_words,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {k: len(v) for k, v in data.items()}
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()