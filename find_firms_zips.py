from pathlib import Path

PROJECT = Path(
    r"C:\Users\DIPANSHU SHUKLA\OneDrive\Desktop\SIH_162_ML"
)

keywords = [
    "DL_FIRE_J1V",
    "DL_FIRE_J2V",
    "DL_FIRE_SV",
    "DL_FIRE_M-C61",
]

print("\nSearching for FIRMS ZIP files...")
print("=" * 70)

found = []

for path in PROJECT.rglob("*.zip"):
    name = path.name

    if any(keyword in name for keyword in keywords):
        found.append(path)

        print("\nFOUND:")
        print(path)
        print("Size:", round(path.stat().st_size / (1024 * 1024), 2), "MB")

print("\n" + "=" * 70)
print("Total matching ZIP files:", len(found))