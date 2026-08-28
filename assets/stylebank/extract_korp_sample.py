#!/usr/bin/env python3
"""Extract Korp sentence samples from a KWIC JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SKIP_TOKENS = {"&", "|"}
PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Korp samples from a KWIC JSON file and output JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/npegl/korp-query.json"),
        help="Input Korp JSON file (default: data/npegl/korp-query.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/npegl/korp_output.json"),
        help="Output JSON file (default: data/npegl/korp_output.json)",
    )
    parser.add_argument(
        "-m",
        "--max-sents",
        type=int,
        default=64,
        help="Maximum number of sentences to keep (default: 64)",
    )
    return parser.parse_args()


def detokenize(words: list[str]) -> str:
    parts: list[str] = []

    for word in words:
        if not word:
            continue
        if not parts:
            parts.append(word)
            continue

        if word in PUNCT_NO_SPACE_BEFORE:
            parts[-1] = parts[-1] + word
        elif parts[-1] in PUNCT_NO_SPACE_AFTER:
            parts[-1] = parts[-1] + word
        else:
            parts.append(word)

    sentence = " ".join(parts)
    sentence = " ".join(sentence.split())
    return sentence


def clean_token(word: str) -> str:
    token = word.strip()
    if not token:
        return ""

    if token == ".":
        return token

    start = 0
    end = len(token)

    while start < end and not (token[start].isalpha() or token[start].isdigit()):
        start += 1
    while end > start and not (token[end - 1].isalpha() or token[end - 1].isdigit()):
        end -= 1

    cleaned = token[start:end]
    if not cleaned or not any(ch.isalpha() for ch in cleaned):
        return ""

    return cleaned


def extract_samples(input_path: Path, max_sents: int) -> list[str]:
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    samples: set[str] = set()
    for entry in data.get("kwic", []):
        tokens = entry.get("tokens", [])
        words = []

        for token in tokens:
            word = (token.get("word") or "").strip()
            if not word or word in SKIP_TOKENS:
                continue

            cleaned = clean_token(word)
            if not cleaned:
                continue
            words.append(cleaned)

        sentence = detokenize(words)
        if not sentence.endswith("."):
            continue

        samples.add(sentence)
        if len(samples) >= max_sents:
            break

    return sorted(list(samples))


def main() -> None:
    args = parse_args()
    sentences = extract_samples(input_path=args.input, max_sents=args.max_sents)

    output = {"old_english": sentences}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"old_english": len(sentences)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()