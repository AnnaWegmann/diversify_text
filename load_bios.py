import pandas as pd

df = pd.read_csv("data/bios_400.csv", usecols=["id", "bio"])
df["bio"] = df["bio"].str.strip()

print(f"Loaded {len(df)} bios")
print(df.head())
