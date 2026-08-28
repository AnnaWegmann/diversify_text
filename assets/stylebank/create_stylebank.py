"""
[Language variation
  [Individual
    [Idiolect
      [Rihanna]
      [Shakira]
      [...]
    ]
  ]
  [Intra-group
    [Diachronic
      [Old English]
      [Middle English]
      [...]
    ]
    [Diatopic
      [Indian English]
      [Kenyan English]
      [...]
    ]
    [Diastratic
      [Age: 55--74]
      [Education: Bachelor]
      [...]
    ]
    [Diaphasic
      [Informational]
      [Opinion]
      [...]
    ]
    [Diamesic
      [Digital]
      [Spoken]
    ]
  ]
]

Scan dataset JSON outputs under ``data/`` and combine them into a
single "stylebank" JSON following the structure above.

Usage:
  python create_stylebank.py --root data --output stylebank.json

The output is a single JSON file (default: ``stylebank.json``).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_output_jsons(root: Path) -> List[Path]:
    results: List[Path] = []
    for p in root.rglob("*_output.json"):
        if p.is_file():
            results.append(p)
    return sorted(results)


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def build_stylebank(paths: List[Path]) -> Dict[str, Any]:

    """
    Build the stylebank structure from the dataset JSON outputs.
    Diachronic: from: chaucer, ept, korp, tcp
    Diatopic: from: ewave
    Diastratic: from: pastel
    Diaphasic: from core
    Diamesic: from: reddit, tal
    Individual / Idiolect: from: tweets
    """

    bank: Dict[str, Dict[str, Dict[str, List[str]]]] = {
        "language_variation": {

            "individual": {
                "idiolect": {}
                },

            "intra-group": {
                "diachronic": {},
                "diatopic": {},
                "diastratic": {},
                "diaphasic": {},
                "diamesic": {},
            },

        }
    }

    diachronic_jsons = ["chaucer", "ept", "korp", "tcp"]
    diatopic_jsons = ["ewave"]
    diastratic_jsons = ["pastel"]
    diaphasic_jsons = ["core"]
    diamesic_jsons = ["reddit", "tal"]
    idiolect_jsons = ["tweets"]

    # open jsons and add to stylebank
    for path in paths:
        data = load_json(path)
        if not data:
            continue

        name = path.stem.replace("_output", "")

        # determine branch container
        if name in diachronic_jsons:
            branch_container = bank["language_variation"]["intra-group"]["diachronic"]
        elif name in diatopic_jsons:
            branch_container = bank["language_variation"]["intra-group"]["diatopic"]
        elif name in diastratic_jsons:
            branch_container = bank["language_variation"]["intra-group"]["diastratic"]
        elif name in diaphasic_jsons:
            branch_container = bank["language_variation"]["intra-group"]["diaphasic"]
        elif name in diamesic_jsons:
            branch_container = bank["language_variation"]["intra-group"]["diamesic"]
        elif name in idiolect_jsons:
            branch_container = bank["language_variation"]["individual"]["idiolect"]

        for key, val in data.items():
                key = key.strip().lower().replace(" ", "_")
                branch_container[key] = val

    return bank


def write_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create stylebank JSON from dataset outputs")
    parser.add_argument("--root", default="data", help="root data directory to scan")
    parser.add_argument("--output", default="stylebank.json", help="output JSON file path")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.output)

    if not root.exists() or not root.is_dir():
        print(f"Data root not found: {root}")
        return 2

    paths = find_output_jsons(root)
    if not paths:
        print(f"No *_output.json files found under {root}")
        return 1

    bank = build_stylebank(paths)
    write_json(bank, out)
    print(f"Wrote stylebank with {len(paths)} files to {out}")
    return 0


if __name__ == "__main__":
    main()
