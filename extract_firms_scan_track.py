from pathlib import Path
from zipfile import ZipFile
import pandas as pd

RAW_DIR = Path(r"data/raw_data")
OUTPUT_DIR = Path(r"data/processed_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

zip_files = sorted(RAW_DIR.glob("DL_FIRE_*.zip"))

print("\nEXTRACTING ORIGINAL FIRMS DATA")
print("=" * 80)

all_data = []

for zip_path in zip_files:

    print(f"\nProcessing: {zip_path.name}")

    with ZipFile(zip_path, "r") as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            print("  No CSV found.")
            continue

        csv_name = csv_files[0]

        print(f"  CSV: {csv_name}")

        with z.open(csv_name) as f:

            df = pd.read_csv(f)

        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

        print("  Checking scan/track...")

        if "scan" in df.columns:
            print("  ✓ scan found")
        else:
            print("  ✗ scan missing")

        if "track" in df.columns:
            print("  ✓ track found")
        else:
            print("  ✗ track missing")

        print("\n  Columns:")
        print("  ", list(df.columns))

        # Keep source archive information
        df["source_archive"] = zip_path.name

        all_data.append(df)


print("\n" + "=" * 80)

if not all_data:
    raise RuntimeError("No FIRMS CSV files found.")


combined = pd.concat(
    all_data,
    ignore_index=True
)

output_file = (
    OUTPUT_DIR /
    "firms_original_with_scan_track.csv"
)

combined.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved: {output_file}"
)

print(
    f"Total rows: {len(combined):,}"
)

print(
    f"Total columns: {len(combined.columns)}"
)

print("\nFinal columns:")
for col in combined.columns:
    print(" -", col)