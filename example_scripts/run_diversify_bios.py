"""Run diversify over the full bios dataset and write JSONL output.

Example:
    python example_scripts/run_diversify_bios.py --methods tinystyler --n-styles 3
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from diversify import Diversifier
from utils.load_bios import load_bios


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method must be provided.")
    return methods


def row_to_record(row: pd.Series) -> dict:
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


def main() -> None:
    script_dir = Path(__file__).resolve().parent  # dynamically determine script directory for relative paths
    default_input = script_dir / "data" / "bios_400.csv"
    default_output = script_dir / "data" / "bios_400_diversified.jsonl"

    parser = argparse.ArgumentParser(
        description="Diversify all bios in a CSV and save JSONL results."
    )
    parser.add_argument("--input", default=str(default_input), help="Input CSV path.")
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--text-column",
        default="bio",
        help="CSV column containing source text to diversify.",
    )
    parser.add_argument(
        "--methods",
        default="tinystyler",
        help="Comma-separated method names (e.g. 'tinystyler,echo').",
    )
    parser.add_argument("--n-styles", type=int, default=3, help="Paraphrases per text.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=None, help="Torch device (cpu/cuda/mps).")
    parser.add_argument(
        "--strict-methods",
        action="store_true",
        help="Raise immediately when a method fails instead of falling back.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_bios(path=input_path, text_column=args.text_column)

    methods = parse_methods(args.methods)
    diversifier = Diversifier(
        device=args.device,
        methods=methods,
        strict_methods=args.strict_methods,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(df), args.batch_size):
            end = min(start + args.batch_size, len(df))
            batch_df = df.iloc[start:end]
            texts = batch_df[args.text_column].fillna("").astype(str).tolist()

            results = diversifier.diversify(
                texts,
                n_styles=args.n_styles,
                text_column=args.text_column,
            )

            for (_, row), result in zip(batch_df.iterrows(), results):
                record = row_to_record(row)
                record["paraphrases"] = result["paraphrases"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Processed {end}/{len(df)}")

    print(f"Wrote diversified bios to: {output_path}")


if __name__ == "__main__":
    main()
