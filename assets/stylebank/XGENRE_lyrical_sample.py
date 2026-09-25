"""Sample 64 CORE "Prose/Lyrical" texts from X-GENRE.

The train split alone only has 47 such rows (dev 15, test 16), so all three
splits are pooled (78 rows) before sampling.
"""

import json
import random

from datasets import concatenate_datasets, load_dataset

REPO = "TajaKuzmanPungersek/X-GENRE-text-genre-dataset"

parts = []
for cfg in ("train", "dev", "test"):  # configs; each has a single "train" split
    ds = load_dataset(REPO, cfg, split="train")
    parts.append(ds.add_column("split", [cfg] * len(ds)))

pool = concatenate_datasets(parts).filter(
    lambda r: r["dataset"] == "CORE" and r["labels"] == "Prose/Lyrical"
)

print(f"{len(pool)} CORE Prose/Lyrical rows available")
rows = list(pool)

with open("core_prose_lyrical_64.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows to core_prose_lyrical_64.jsonl")