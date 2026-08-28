#!/usr/bin/env python3
"""Extract sentence samples from the english-philosophical-texts repository."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/earlytexts/english-philosophical-texts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Early Modern English samples from english-philosophical-texts."
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git repository URL (default: earlytexts/english-philosophical-texts)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/ept/ept_output.json"),
        help="Output JSON file (default: data/ept/ept_output.json)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=5,
        help="Number of random files to sample deterministically (default: 5)",
    )
    parser.add_argument(
        "-m",
        "--max-sents",
        type=int,
        default=64,
        help="Maximum number of sentences to keep (default: 64)",
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=100,
        help="Number of lines to inspect around each file center (default: 100)",
    )
    parser.add_argument(
        "-w",
        "--min-words",
        type=int,
        default=10,
        help="Minimum words per sentence (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per sentence (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    return parser.parse_args()


def run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def clone_repo(repo_url: str, workdir: Path) -> Path:
    repo_dir = workdir / "philosophical_repo"
    run_git(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            repo_url,
            str(repo_dir),
        ]
    )
    return repo_dir


def list_mit_files(repo_dir: Path) -> list[str]:
    output = run_git(["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD", "texts"])
    all_mit = [line.strip() for line in output.splitlines() if line.strip().endswith(".mit")]

    # Prefer non-index files because index.mit entries are often metadata-only wrappers.
    non_index = [path for path in all_mit if not path.endswith("/index.mit")]
    files = non_index if non_index else all_mit
    return sorted(files)


def read_file(repo_dir: Path, path: str) -> str:
    return run_git(["git", "-C", str(repo_dir), "show", f"HEAD:{path}"])


def starts_with_capital(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.isupper()
    return False


def normalize_line(line: str) -> str:
    cleaned = line.strip()

    # Strip common Markit inline markers.
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = cleaned.replace("£", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("=", "")
    cleaned = re.sub(r"[\^|#~]", "", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def sentence_candidates(text: str, center_window: int, min_words: int, max_tokens: int) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []

    center = len(lines) // 2
    half_window = max(1, center_window // 2)
    start = max(0, center - half_window)
    end = min(len(lines), center + half_window + 1)

    center_lines = [normalize_line(raw_line) for raw_line in lines[start:end]]
    center_lines = [line for line in center_lines if line]
    block = " ".join(center_lines)
    block = " ".join(block.split())

    # Simple sentence splitting that keeps period-terminated sentences.
    rough_sentences = re.split(r"(?<=\.)\s+", block)

    candidates: list[str] = []
    for sentence in rough_sentences:
        line = sentence.strip()
        if not line:
            continue
        if not line.endswith("."):
            continue
        if len(line.split()) < min_words:
            continue
        if len(line.split()) > max_tokens:
            continue
        if not any(ch.isalpha() for ch in line):
            continue
        if not starts_with_capital(line):
            continue
        if any(ch.isdigit() for ch in line):
            continue
        candidates.append(line)

    return candidates


def extract_samples(
    repo_url: str,
    num_files: int,
    max_sents: int,
    center_window: int,
    min_words: int,
    max_tokens: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_repo(repo_url, Path(tmpdir))
        mit_files = list_mit_files(repo_dir)
        if not mit_files:
            return []

        chosen_files = rng.sample(mit_files, min(num_files, len(mit_files)))
        per_file_target = max(1, max_sents // max(1, len(chosen_files)))

        samples: list[str] = []
        for file_path in chosen_files:
            try:
                content = read_file(repo_dir, file_path)
            except subprocess.CalledProcessError:
                continue

            candidates = sentence_candidates(
                content,
                center_window=center_window,
                min_words=min_words,
                max_tokens=max_tokens,
            )
            if not candidates:
                continue

            if len(candidates) > per_file_target:
                chosen_indices = sorted(rng.sample(range(len(candidates)), per_file_target))
                candidates = [candidates[i] for i in chosen_indices]

            samples.extend(candidates)
            if len(samples) >= max_sents:
                break

    return samples[:max_sents]


def main() -> None:
    args = parse_args()
    sentences = extract_samples(
        repo_url=args.repo_url,
        num_files=args.num_files,
        max_sents=args.max_sents,
        center_window=args.center_window,
        min_words=args.min_words,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    output = {"late_modern_english": sentences}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"late_modern_english": len(sentences)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()