"""Quick smoke test: style-transfer the first 2 bios into an informal style."""

import pandas as pd
from diversify.method.tinystyler.model import TinyStyler

# --- Load first 2 bios ------------------------------------------------
df = pd.read_csv("example_scripts/data/bios_400.csv", usecols=["id", "bio"])
df["bio"] = df["bio"].str.strip()
bios = df.head(2)

texts = bios["bio"].tolist()

# --- 5 informal style examples -----------------------------------------
informal_examples = [
    "lol he just showed up outta nowhere and everyone lost it",
    "she's super into painting, like you wouldn't even believe how good she is",
    "dude went to college in texas or whatever and ended up working at some tech company",
    "tbh i have no clue where he's from but he seems pretty chill",
    "she quit her job and moved across the country, kinda wild ngl",
]

# --- Run transfer -------------------------------------------------------
print("Loading TinyStyler model...")
ts = TinyStyler()
print(f"Model loaded (device={ts.device})\n")

print("Transferring 2 bios to informal style...\n")
results = ts.transfer(texts, style=informal_examples)

for idx, (original, transferred) in enumerate(zip(texts, results)):
    bio_id = bios.iloc[idx]["id"]
    print(f"=== Bio {bio_id} ===")
    print(f"ORIGINAL:    {original}")
    print(f"TRANSFERRED: {transferred}")
    print()
