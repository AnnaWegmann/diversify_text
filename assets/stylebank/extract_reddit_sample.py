#!/usr/bin/env python3
"""Extract cleaned Reddit comment samples from the Reddit_Post_Comment_Dataset repo."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/ishandandekar/Reddit_Post_Comment_Dataset"
COMMENT_PREFIX = "###### "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Reddit comments from Reddit_Post_Comment_Dataset and output JSON."
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git repository URL (default: ishandandekar/Reddit_Post_Comment_Dataset)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/reddit/reddit_output.json"),
        help="Output JSON file (default: data/reddit/reddit_output.json)",
    )
    parser.add_argument(
        "-m",
        "--max-samples",
        type=int,
        default=64,
        help="Maximum comments to keep (default: 64)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum tokens per comment (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per comment (default: 100)",
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
        text=False,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def clone_repo(repo_url: str, workdir: Path) -> Path:
    repo_dir = workdir / "reddit_repo"
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


def list_data_files(repo_dir: Path) -> list[str]:
    output = run_git(["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD", "data"])
    files = [line.strip() for line in output.splitlines() if line.strip().endswith(".txt")]
    return sorted(files)


def read_file(repo_dir: Path, path: str) -> str:
    return run_git(["git", "-C", str(repo_dir), "show", f"HEAD:{path}"])


def decode_comment_literal(raw: str) -> str:
    payload = raw.strip()

    try:
        if payload.startswith("b'") or payload.startswith('b"'):
            value = ast.literal_eval(payload)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                return value

        # Fallback if the data line is quoted text without byte-literal prefix.
        value = ast.literal_eval(payload)
        if isinstance(value, str):
            return value
    except Exception:
        pass

    return payload


def normalize(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("\r", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\^+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_bad_comment(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return True
    if lower in {"[deleted]", "[removed]", "deleted", "removed"}:
        return True
    if re.search(r"https?://\S+|www\.\S+|\breddit\.com\b", text, flags=re.IGNORECASE):
        return True
    return False


def extract_comments(
    repo_url: str,
    max_samples: int,
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = clone_repo(repo_url, Path(tmpdir))
        files = list_data_files(repo_dir)
        if not files:
            return []

        comments: list[str] = []
        for path in files:
            content = read_file(repo_dir, path)
            for line in content.splitlines():
                if not line.startswith(COMMENT_PREFIX):
                    continue

                literal = line[len(COMMENT_PREFIX) :].strip()
                comment = decode_comment_literal(literal)
                comment = normalize(comment)
                if is_bad_comment(comment):
                    continue

                token_count = len(comment.split())
                if token_count < min_tokens or token_count > max_tokens:
                    continue

                comments.append(comment)

    # Deduplicate while preserving encounter order, then sample deterministically.
    unique_comments = list(dict.fromkeys(comments))
    if not unique_comments:
        return []

    rng = random.Random(seed)
    if len(unique_comments) > max_samples:
        chosen = rng.sample(unique_comments, max_samples)
    else:
        chosen = unique_comments
    return chosen


def main() -> None:
    args = parse_args()
    comments = extract_comments(
        repo_url=args.repo_url,
        max_samples=args.max_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    output = {"digital_communication": comments}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"digital_communication": len(comments)}, indent=2, ensure_ascii=False))
    print(f"Wrote samples to: {args.output}")


if __name__ == "__main__":
    main()