#!/usr/bin/env python3
"""Compute summary statistics for a stylebank JSON file.

The script walks the nested stylebank structure, finds every leaf list of
sentences, and reports:
- total number of leaf categories
- number of leaf categories per top-level branch
- sentence counts per leaf category
- sentence length stats per leaf category and per branch

Usage:
  python stylebank_stats.py --input stylebank.json
  python stylebank_stats.py --input stylebank.json --output stylebank_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stats for a stylebank JSON file")
    parser.add_argument("--input", type=Path, default=Path("stylebank.json"), help="Stylebank JSON file")
    parser.add_argument("--output", type=Path, help="Optional path to write the stats JSON")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_sentence_leaf(value: Any) -> bool:
    return isinstance(value, list) and (not value or all(isinstance(item, str) for item in value))


def sentence_len(sentence: str) -> int:
    return len(sentence.split())


def summarise_numbers(values: Iterable[int]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {"min": min(values), "max": max(values), "avg": mean(values)}


def format_number(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def make_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in string_rows)
    return "\n".join(lines)


def print_summary(stats: dict[str, Any]) -> None:
    overall_rows = [[
        "num_leafs",
        stats["num_leafs"],
        "sentences/leaf avg",
        format_number(stats["sentences_per_leaf"]["avg"]),
        "sentence words avg",
        format_number(stats["sentence_length_words"]["avg"]),
    ]]
    print("Overall")
    print(make_table(["metric", "value", "metric", "value", "metric", "value"], overall_rows))
    print()

    category_rows = [
        [category, item["num_leafs"]]
        for category, item in sorted(stats["per_category"].items())
    ]
    print("Leafs per category")
    print(make_table(["category", "num_leafs"], category_rows))
    print()

    branch_rows = [
        [branch, item["num_leafs"], format_number(item["sentences_per_leaf"]["avg"]), format_number(item["sentence_length_words"]["avg"])]
        for branch, item in sorted(stats["per_branch"].items())
    ]
    print("Leafs per group")
    print(make_table(["group", "num_leafs", "sentences/leaf avg", "sentence words avg"], branch_rows))


def walk_leaves(node: Any, path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []

    if is_sentence_leaf(node):
        leaves.append({"path": path, "sentences": list(node)})
        return leaves

    if isinstance(node, dict):
        for key, value in node.items():
            leaves.extend(walk_leaves(value, path + (str(key),)))

    return leaves


def top_branch(path: tuple[str, ...]) -> str:
    if len(path) >= 2 and path[0] == "language_variation":
        return path[1]
    if path:
        return path[0]
    return "<root>"


def style_category(path: tuple[str, ...]) -> str:
    if len(path) >= 3 and path[0] == "language_variation":
        return path[2]
    if path:
        return path[-1]
    return "<root>"


def leaf_name(path: tuple[str, ...]) -> str:
    return path[-1] if path else "<root>"


def build_stats(data: Any) -> dict[str, Any]:
    leaves = walk_leaves(data)

    leaf_stats: list[dict[str, Any]] = []
    leaf_counts_by_branch: dict[str, int] = {}
    leaf_counts_by_category: dict[str, int] = {}
    leaf_paths_by_branch: dict[str, list[str]] = {}
    leaf_paths_by_category: dict[str, list[str]] = {}

    for leaf in leaves:
        path = tuple(leaf["path"])
        sentences = leaf["sentences"]
        branch = top_branch(path)
        category = style_category(path)
        name = leaf_name(path)
        lengths = [sentence_len(sentence) for sentence in sentences]

        leaf_stats.append(
            {
                "path": list(path),
                "branch": branch,
                "category": category,
                "leaf": name,
                "num_sentences": len(sentences),
                "sentence_length_words": summarise_numbers(lengths),
            }
        )

        leaf_counts_by_branch[branch] = leaf_counts_by_branch.get(branch, 0) + 1
        leaf_counts_by_category[category] = leaf_counts_by_category.get(category, 0) + 1
        leaf_paths_by_branch.setdefault(branch, []).append("/".join(path))
        leaf_paths_by_category.setdefault(category, []).append("/".join(path))

    sentences_per_leaf = [item["num_sentences"] for item in leaf_stats]
    all_sentence_lengths = [
        sentence_len(sentence)
        for leaf in leaves
        for sentence in leaf["sentences"]
    ]

    branch_sentence_lengths: dict[str, list[int]] = {}
    branch_sentences_per_leaf: dict[str, list[int]] = {}
    for item in leaf_stats:
        branch = item["branch"]
        branch_sentence_lengths.setdefault(branch, [])
        branch_sentences_per_leaf.setdefault(branch, [])
        branch_sentences_per_leaf[branch].append(item["num_sentences"])

    for leaf in leaves:
        branch = top_branch(tuple(leaf["path"]))
        branch_sentence_lengths.setdefault(branch, []).extend(
            sentence_len(sentence) for sentence in leaf["sentences"]
        )

    return {
        "input_file": None,
        "num_leafs": len(leaves),
        "num_leafs_per_group": leaf_counts_by_branch,
        "num_leafs_per_category": leaf_counts_by_category,
        "leaf_paths_per_category": leaf_paths_by_branch,
        "leaf_paths_per_style_category": leaf_paths_by_category,
        "sentences_per_leaf": summarise_numbers(sentences_per_leaf),
        "sentence_length_words": summarise_numbers(all_sentence_lengths),
        "per_branch": {
            branch: {
                "num_leafs": leaf_counts_by_branch[branch],
                "sentences_per_leaf": summarise_numbers(branch_sentences_per_leaf.get(branch, [])),
                "sentence_length_words": summarise_numbers(branch_sentence_lengths.get(branch, [])),
            }
            for branch in sorted(leaf_counts_by_branch)
        },
        "per_category": {
            category: {
                "num_leafs": leaf_counts_by_category[category],
            }
            for category in sorted(leaf_counts_by_category)
        },
        "per_leaf": leaf_stats,
    }


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 2

    data = load_json(args.input)
    stats = build_stats(data)
    stats["input_file"] = str(args.input)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote stats to: {args.output}")

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())