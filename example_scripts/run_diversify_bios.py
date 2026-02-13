"""Run diversify over the full bios dataset and write CSV output.

Example:
    python example_scripts/run_diversify_bios.py --methods tinystyler --n-styles 3
"""

import argparse
from pathlib import Path

from diversify import Diversifier
from utils.load_bios import load_bios


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method must be provided.")
    return methods


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "bios_400.csv"
    default_output = script_dir / "data" / "bios_400_diversified.csv"

    parser = argparse.ArgumentParser(
        description="Diversify all bios in a CSV and save CSV results."
    )
    parser.add_argument("--input", default=str(default_input), help="Input CSV path.")
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output CSV path.",
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
    parser.add_argument(
        "--split-on-punctuation",
        action="store_true",
        help="Split each text by punctuation before diversification.",
    )
    parser.add_argument("--device", default=None, help="Torch device (cpu/cuda/mps).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_bios(path=input_path, text_column=args.text_column)

    methods = parse_methods(args.methods)
    diversifier = Diversifier(
        device=args.device,
        methods=methods,
    )

    output_df = diversifier.diversify(
        df,
        n_styles=args.n_styles,
        text_column=args.text_column,
        batch_size=args.batch_size,
        split_on_punctuation=args.split_on_punctuation,
    )
    if not hasattr(output_df, "to_csv"):
        raise TypeError("Expected DataFrame output for DataFrame input.")
    output_df.to_csv(output_path, index=False)
    print(f"Wrote diversified bios CSV to: {output_path}")


if __name__ == "__main__":
    main()
