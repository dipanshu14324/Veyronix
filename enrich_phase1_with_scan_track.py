from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PHASE1_FILE = (
    BASE_DIR
    / "data"
    / "phase1_fire_data"
    / "Phase_1_India_Hotspots_Combined.csv"
)

ORIGINAL_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "firms_original_with_scan_track.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "phase1_firms_with_scan_track.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Phase-1 data...")
phase1 = pd.read_csv(PHASE1_FILE)

print("Phase-1 rows:", len(phase1))


print("\nLoading original FIRMS data...")
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

print("Original FIRMS rows:", len(original))


# ============================================================
# NORMALIZE MATCHING COLUMNS
# ============================================================

print("\nPreparing matching keys...")


def normalize_time(value):
    """
    Convert FIRMS acquisition time into HHMM format.
    """
    if pd.isna(value):
        return ""

    try:
        return f"{int(float(value)):04d}"
    except:
        return str(value).strip()


phase1["acq_date"] = (
    pd.to_datetime(
        phase1["acq_date"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

original["acq_date"] = (
    pd.to_datetime(
        original["acq_date"],
        errors="coerce"
    )
    .dt.strftime("%Y-%m-%d")
)

phase1["acq_time_key"] = phase1["acq_time"].apply(
    normalize_time
)

original["acq_time_key"] = original["acq_time"].apply(
    normalize_time
)


# ============================================================
# CREATE MATCHING KEY
# ============================================================

# Round coordinates slightly to tolerate tiny floating-point
# representation differences.

phase1["lat_key"] = phase1["latitude"].round(5)
phase1["lon_key"] = phase1["longitude"].round(5)

original["lat_key"] = original["latitude"].round(5)
original["lon_key"] = original["longitude"].round(5)


match_columns = [
    "lat_key",
    "lon_key",
    "acq_date",
    "acq_time_key",
    "satellite",
]


print("\nCreating matching key...")

phase1["match_key"] = (
    phase1[match_columns]
    .astype(str)
    .agg("|".join, axis=1)
)

original["match_key"] = (
    original[match_columns]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# CHECK DUPLICATES IN ORIGINAL DATA
# ============================================================

print("\nChecking original FIRMS key duplicates...")

duplicate_count = original["match_key"].duplicated().sum()

print(
    "Duplicate matching keys:",
    duplicate_count
)


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

lookup = original[
    [
        "match_key",
        "scan",
        "track",
        "brightness",
        "frp",
    ]
].copy()


# Remove duplicate keys while preserving the first record.

lookup = lookup.drop_duplicates(
    subset="match_key",
    keep="first"
)


# ============================================================
# MERGE
# ============================================================

print("\nMatching Phase-1 observations...")

enriched = phase1.merge(
    lookup,
    on="match_key",
    how="left",
    suffixes=("", "_original")
)


# ============================================================
# MATCH STATISTICS
# ============================================================

matched = enriched["scan"].notna()

matched_count = matched.sum()
unmatched_count = (~matched).sum()

print("\n" + "=" * 70)
print("MATCH RESULTS")
print("=" * 70)

print(
    f"Phase-1 rows      : {len(enriched):,}"
)

print(
    f"Matched scan/track: {matched_count:,}"
)

print(
    f"Unmatched          : {unmatched_count:,}"
)

print(
    f"Match rate         : {matched.mean() * 100:.2f}%"
)


# ============================================================
# SCAN / TRACK STATISTICS
# ============================================================

if matched_count > 0:

    print("\nSCAN statistics:")
    print(enriched.loc[matched, "scan"].describe())

    print("\nTRACK statistics:")
    print(enriched.loc[matched, "track"].describe())


# ============================================================
# CLEAN TEMPORARY COLUMNS
# ============================================================

enriched = enriched.drop(
    columns=[
        "acq_time_key",
        "lat_key",
        "lon_key",
        "match_key",
    ],
    errors="ignore"
)


# ============================================================
# SAVE
# ============================================================

enriched.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print(
    "\nRows:",
    f"{len(enriched):,}"
)

print(
    "Columns:",
    len(enriched.columns)
)

print("\nNew columns:")
print(" - scan")
print(" - track")

print("\nDone.")