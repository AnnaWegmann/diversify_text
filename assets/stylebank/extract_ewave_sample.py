#!/usr/bin/env python3
"""Extract E-WAVE sentence samples per language from CSV files.

Rules:
- Read language id -> language name mapping from languages.csv.
- Read sentences from examples.csv column Primary_Text, grouped by Language_ID.
- Keep sentences with at least N words.
- Keep up to M sentences per language.
- Output JSON mapping language name to sentence lists.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample E-WAVE examples by language and output JSON."
    )
    parser.add_argument(
        "--languages-csv",
        type=Path,
        default=Path("data/ewave/languages.csv"),
        help="Input languages CSV file path",
    )
    parser.add_argument(
        "--examples-csv",
        type=Path,
        default=Path("data/ewave/examples.csv"),
        help="Input examples CSV file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/ewave/ewave_output.json"),
        help="Output JSON file path (default: data/ewave/ewave_output.json)",
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=64,
        help="Maximum number of sentences per language (default: 64)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum words per sentence (default: 10)",
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


def load_language_mapping(languages_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with languages_csv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lang_id = (row.get("ID") or "").strip()
            lang_name = (row.get("Name") or "").strip()
            if lang_id and lang_name:
                mapping[lang_id] = lang_name

    return mapping


def extract_samples(
    language_mapping: dict[str, str],
    examples_csv: Path,
    max_per_language: int,
    min_words: int,
    seed: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}

    # Some source rows can contain very large text blobs.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    with examples_csv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lang_id = (row.get("Language_ID") or "").strip()
            sentence = (row.get("Primary_Text") or "").strip()

            if not lang_id or not sentence:
                continue

            lang_name = language_mapping.get(lang_id)
            if not lang_name:
                continue

            if word_count(sentence) < min_words:
                continue

            grouped.setdefault(lang_name, []).append(sentence)

    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}

    for lang_name, candidates in grouped.items():
        # Deduplicate sentences while preserving order.
        seen = set()
        unique_candidates = []
        for s in candidates:
            if s not in seen:
                seen.add(s)
                unique_candidates.append(s)
        candidates = unique_candidates
        if len(candidates) > max_per_language:
            candidates = rng.sample(candidates, max_per_language)
        sampled[lang_name] = candidates

        # Remove languages with less than 10 sentences after filtering.
        if len(candidates) < 10:
            del sampled[lang_name]

    # Sort output by language name.
    return dict(sorted(sampled.items()))


def main() -> None:
    args = parse_args()

    mapping = load_language_mapping(args.languages_csv)
    data = extract_samples(
        language_mapping=mapping,
        examples_csv=args.examples_csv,
        max_per_language=args.max_per_language,
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
