#!/usr/bin/env python3
"""Extract core sentence samples per label from a TSV file.

Rules:
- Input is TSV.
- Label is read from the first column (first token in that column).
- Sentence column is auto-detected as the column immediately after the first
  column that contains a numeric id.
- Keep sentences with 10-100 words.
- Sample up to N sentences per requested label.
- Output JSON mapping style keys to sentence lists.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


TARGET_LABELS = ["IN", "OP", "NA", "IP", "ID", "HI", "LY"]

LABEL_TO_OUTPUT_KEY = {
	"IN": "informational",
	"OP": "opinion",
	"NA": "narrative",
	"IP": "persuasive",
	"ID": "interactive",
	"HI": "instructional",
	"LY": "lyrical",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Sample core sentences by label from TSV and output JSON."
	)
	parser.add_argument("--tsv", type=Path, default=Path("data/core/train.tsv"), help="Input TSV file path")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path("data/core/core_output.json"),
		help="Output JSON file path (default: data/core/core_output.json)",
	)
	parser.add_argument(
		"--max-per-label",
		type=int,
		default=64,
		help="Maximum number of sentences per label (default: 64)",
	)
	parser.add_argument(
		"--min-words",
		type=int,
		default=10,
		help="Minimum words per sentence (default: 10)",
	)
	parser.add_argument(
		"--max-words",
		type=int,
		default=100,
		help="Maximum words per sentence (default: 100)",
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


def extract_label(label_cell: str) -> str:
	# Some datasets store extra tokens in column 0 (e.g. "IN OI").
	return label_cell.strip().split()[0] if label_cell.strip() else ""


def extract_samples(
	tsv_path: Path,
	max_per_label: int,
	min_words: int,
	max_words: int,
	seed: int,
) -> dict[str, list[str]]:
	grouped: dict[str, list[str]] = {label: [] for label in TARGET_LABELS}

	# Some source rows can contain very large text blobs.
	try:
		csv.field_size_limit(sys.maxsize)
	except OverflowError:
		csv.field_size_limit(2**31 - 1)

	with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
		reader = csv.reader(fh, delimiter="\t")
		for row in reader:
			if len(row) < 3:
				continue

			label = extract_label(row[0])
			if label not in grouped:
				continue

			id_index = None
			for i, cell in enumerate(row):
				if cell.strip().isdigit():
					id_index = i
					break

			if id_index is None or id_index + 1 >= len(row):
				continue

			sentence = row[id_index + 1].strip()
			if not sentence:
				continue

			if not sentence.endswith("."):
				sentence = sentence.rstrip("!?;:,")
				sentence = sentence.rstrip()
				sentence = f"{sentence}."

			wc = word_count(sentence)
			if min_words <= wc <= max_words:
				grouped[label].append(sentence)

	rng = random.Random(seed)
	sampled: dict[str, list[str]] = {}
	for label in TARGET_LABELS:
		candidates = grouped[label]
		if len(candidates) > max_per_label:
			candidates = rng.sample(candidates, max_per_label)
		if len(candidates) >= 10:
			sampled[LABEL_TO_OUTPUT_KEY[label]] = candidates

	return sampled


def main() -> None:
	args = parse_args()
	data = extract_samples(
		tsv_path=args.tsv,
		max_per_label=args.max_per_label,
		min_words=args.min_words,
		max_words=args.max_words,
		seed=args.seed,
	)

	# Sort output by key name
	data = dict(sorted(data.items()))

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

	counts = {k: len(v) for k, v in data.items()}
	print(json.dumps(counts, indent=2))
	print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
	main()
