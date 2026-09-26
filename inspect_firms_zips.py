from pathlib import Path
from zipfile import ZipFile

RAW_DIR = Path(
    r"data/raw_data"
)

zip_files = sorted(RAW_DIR.glob("DL_FIRE_*.zip"))

print("\nFIRMS ZIP CONTENTS")
print("=" * 80)

for zip_path in zip_files:

    print(f"\nZIP: {zip_path.name}")
    print("-" * 80)

    try:
        with ZipFile(zip_path, "r") as z:

            files = z.namelist()

            for file in files:
                print("  ", file)

    except Exception as e:
        print("ERROR:", e)