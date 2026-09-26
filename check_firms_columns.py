import pandas as pd

FILE = r"data/phase1_fire_data/Phase_1_India_Hotspots_Combined.csv"

df = pd.read_csv(FILE, nrows=5)

print("Columns:")
for col in df.columns:
    print(" -", col)

print("\nShape of sample:", df.shape)

required = ["scan", "track"]

print("\nChecking required columns:")

for col in required:
    if col in df.columns:
        print(f"✓ {col} is present")
    else:
        print(f"✗ {col} is MISSING")