from pathlib import Path
import pandas as pd
import numpy as np

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
# LOAD
# ============================================================

print("\nLoading Phase-1...")
phase1 = pd.read_csv(PHASE1_FILE)

print("Phase-1:", len(phase1))


print("\nLoading original FIRMS...")
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

print("Original:", len(original))


# ============================================================
# NORMALIZE SATELLITE
# ============================================================

def normalize_satellite(x):

    x = str(x).strip()

    mapping = {
        "NOAA-20 VIIRS": "N20",
        "NOAA-21 VIIRS": "N21",
        "Suomi-NPP VIIRS": "SNPP",
        "Terra MODIS": "Terra",
        "Aqua MODIS": "Aqua",
        "N20": "N20",
        "N21": "N21",
        "SNPP": "SNPP",
        "Terra": "Terra",
        "Aqua": "Aqua",
    }

    return mapping.get(x, x)


phase1["satellite_key"] = (
    phase1["satellite"]
    .apply(normalize_satellite)
)

original["satellite_key"] = (
    original["satellite"]
    .apply(normalize_satellite)
)


# ============================================================
# NORMALIZE TIME
# ============================================================

phase1["acq_time_key"] = (
    phase1["acq_time"]
    .astype(int)
)

original["acq_time_key"] = (
    original["acq_time"]
    .astype(int)
)


# ============================================================
# NORMALIZE DATE
# ============================================================

phase1["acq_date_key"] = (
    pd.to_datetime(
        phase1["acq_date"]
    ).dt.strftime("%Y-%m-%d")
)

original["acq_date_key"] = (
    pd.to_datetime(
        original["acq_date"]
    ).dt.strftime("%Y-%m-%d")
)


# ============================================================
# EXACT COORDINATE KEY
# ============================================================

phase1["lat_key"] = phase1["latitude"].round(5)
phase1["lon_key"] = phase1["longitude"].round(5)

original["lat_key"] = original["latitude"].round(5)
original["lon_key"] = original["longitude"].round(5)


# ============================================================
# MATCH KEY
# ============================================================

keys = [
    "lat_key",
    "lon_key",
    "acq_date_key",
    "acq_time_key",
    "satellite_key",
]

phase1["match_key"] = (
    phase1[keys]
    .astype(str)
    .agg("|".join, axis=1)
)

original["match_key"] = (
    original[keys]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# LOOKUP
# ============================================================

lookup = original[
    [
        "match_key",
        "scan",
        "track",
    ]
].copy()


print("\nOriginal duplicate keys:")

print(
    lookup["match_key"]
    .duplicated()
    .sum()
)


lookup = lookup.drop_duplicates(
    "match_key",
    keep="first"
)


# ============================================================
# MERGE
# ============================================================

print("\nMatching...")

result = phase1.merge(
    lookup,
    on="match_key",
    how="left"
)


# ============================================================
# RESULTS
# ============================================================

matched = result["scan"].notna()

matched_count = matched.sum()
unmatched_count = (~matched).sum()

print("\n" + "=" * 70)
print("MATCH RESULTS")
print("=" * 70)

print(
    f"Phase-1 rows       : {len(result):,}"
)

print(
    f"Matched            : {matched_count:,}"
)

print(
    f"Unmatched          : {unmatched_count:,}"
)

print(
    f"Match rate         : {matched.mean() * 100:.2f}%"
)


# ============================================================
# STATISTICS
# ============================================================

if matched_count > 0:

    print("\nSCAN:")
    print(
        result.loc[matched, "scan"]
        .describe()
    )

    print("\nTRACK:")
    print(
        result.loc[matched, "track"]
        .describe()
    )


# ============================================================
# REMOVE TEMP COLUMNS
# ============================================================

result.drop(
    columns=[
        "satellite_key",
        "acq_time_key",
        "acq_date_key",
        "lat_key",
        "lon_key",
        "match_key",
    ],
    inplace=True,
    errors="ignore"
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print("\nColumns:")
print(result.columns.tolist())