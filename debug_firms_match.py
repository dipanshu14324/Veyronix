from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

PHASE1_FILE = (
    BASE_DIR / "data" / "phase1_fire_data"
    / "Phase_1_India_Hotspots_Combined.csv"
)

ORIGINAL_FILE = (
    BASE_DIR / "data" / "processed_data"
    / "firms_original_with_scan_track.csv"
)

phase1 = pd.read_csv(PHASE1_FILE)

original = pd.read_csv(
    ORIGINAL_FILE,
    usecols=[
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "frp",
    ]
)

print("\nPHASE-1 SAMPLE")
print("=" * 80)
print(phase1.head(5).to_string())

print("\n\nORIGINAL FIRMS SAMPLE")
print("=" * 80)
print(original.head(5).to_string())


print("\n\nDATA TYPES")
print("=" * 80)

print("\nPHASE-1:")
print(phase1.dtypes)

print("\nORIGINAL:")
print(original.dtypes)


print("\n\nUNIQUE SATELLITES")
print("=" * 80)

print("Phase-1:")
print(phase1["satellite"].value_counts().head(20))

print("\nOriginal:")
print(original["satellite"].value_counts().head(20))


print("\n\nACQ TIME SAMPLES")
print("=" * 80)

print("Phase-1:")
print(
    phase1["acq_time"]
    .dropna()
    .astype(str)
    .head(20)
    .tolist()
)

print("\nOriginal:")
print(
    original["acq_time"]
    .dropna()
    .astype(str)
    .head(20)
    .tolist()
)


print("\n\nDATE SAMPLES")
print("=" * 80)

print("Phase-1:")
print(
    phase1["acq_date"]
    .head(10)
    .tolist()
)

print("\nOriginal:")
print(
    original["acq_date"]
    .head(10)
    .tolist()
)


print("\n\nCOORDINATE SAMPLES")
print("=" * 80)

print("Phase-1:")
print(
    phase1[
        ["latitude", "longitude"]
    ].head(10).to_string(index=False)
)

print("\nOriginal:")
print(
    original[
        ["latitude", "longitude"]
    ].head(10).to_string(index=False)
)


print("\n\nFRP SAMPLES")
print("=" * 80)

print("Phase-1:")
print(
    phase1["frp"]
    .head(10)
    .tolist()
)

print("\nOriginal:")
print(
    original["frp"]
    .head(10)
    .tolist()
)