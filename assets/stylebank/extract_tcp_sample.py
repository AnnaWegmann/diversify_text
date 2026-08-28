#!/usr/bin/env python3
"""Extract TCP sentence samples from the VEP2_TCP_SimpleText repository."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/uwgraphics/VEP2_TCP_SimpleText"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Early Modern English samples from the TCP SimpleText repository."
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git repository URL for TCP SimpleText (default: GitHub repo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/tcp/tcp_output.json"),
        help="Output JSON file (default: data/tcp/tcp_output.json)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=60,
        help="Number of random files to sample deterministically (default: 60)",
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
        default=300,
        help="Number of lines to inspect around the file center (default: 300)",
    )
    parser.add_argument(
        "-w",
        "--min-words",
        type=int,
        default=10,
        help="Minimum words per sentence (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic file and sentence sampling (default: 42)",
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
    repo_dir = workdir / "tcp_repo"
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


def list_txt_files(repo_dir: Path) -> list[str]:
    output = run_git(["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD"])
    files = [line.strip() for line in output.splitlines() if line.strip().endswith(".txt")]
    return sorted(files)


def read_file(repo_dir: Path, path: str) -> str:
    return run_git(["git", "-C", str(repo_dir), "show", f"HEAD:{path}"])


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def starts_with_capital(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            return ch.isupper()
    return False


def sentence_candidates(text: str, center_window: int, min_words: int) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []

    center = len(lines) // 2
    half_window = max(1, center_window // 2)
    start = max(0, center - half_window)
    end = min(len(lines), center + half_window + 1)

    candidates: list[str] = []
    for raw_line in lines[start:end]:
        line = normalize_line(raw_line)
        if not line:
            continue
        if not line.endswith("."):
            continue
        if len(line.split()) < min_words:
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
    seed: int,
) -> list[str]:
    rng = random.Random(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        repo_dir = clone_repo(repo_url, workdir)
        txt_files = list_txt_files(repo_dir)
        if not txt_files:
            return []

        chosen_files = rng.sample(txt_files, min(num_files, len(txt_files)))
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
        seed=args.seed,
    )

    output = {"early_modern_english": sentences}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"early_modern_english": len(sentences)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()