import pandas as pd
from pathlib import Path

FILE = Path(
    r"data/phase1_fire_data/Phase_1_India_Hotspots_Combined.csv"
)

df = pd.read_csv(
    FILE,
    usecols=["source_file"]
)

print("\nUnique source files:")
print("=" * 60)

files = df["source_file"].dropna().unique()

for file in files:
    print(file)

print("\nTotal unique source files:", len(files))