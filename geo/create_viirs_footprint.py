from pathlib import Path
from math import cos, radians

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "phase1_firms_with_scan_track.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "phase1_viirs_footprints.csv"
)


# ============================================================
# LOAD
# ============================================================

print("\nLoading enriched FIRMS data...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# KEEP VALID FOOTPRINTS
# ============================================================

valid = (
    df["scan"].notna()
    & df["track"].notna()
    & (df["scan"] > 0)
    & (df["track"] > 0)
)

footprints = df.loc[valid].copy()

print(
    f"Valid scan/track rows: {len(footprints):,}"
)


# ============================================================
# APPROXIMATE FOOTPRINT
# ============================================================

# scan and track are treated as approximate pixel dimensions
# in kilometres for this prototype.

meters_per_degree_lat = 111320.0

half_scan_m = (
    footprints["scan"] * 1000 / 2
)

half_track_m = (
    footprints["track"] * 1000 / 2
)

lat_offset = (
    half_track_m /
    meters_per_degree_lat
)

meters_per_degree_lon = (
    meters_per_degree_lat
    * footprints["latitude"]
        .apply(lambda x: cos(radians(x)))
)

lon_offset = (
    half_scan_m /
    meters_per_degree_lon
)


# ============================================================
# BOUNDING BOX
# ============================================================

footprints["min_lat"] = (
    footprints["latitude"] - lat_offset
)

footprints["max_lat"] = (
    footprints["latitude"] + lat_offset
)

footprints["min_lon"] = (
    footprints["longitude"] - lon_offset
)

footprints["max_lon"] = (
    footprints["longitude"] + lon_offset
)


# ============================================================
# FOOTPRINT AREA
# ============================================================

footprints["footprint_area_km2"] = (
    footprints["scan"]
    * footprints["track"]
)


# ============================================================
# FOOTPRINT TYPE
# ============================================================

footprints["footprint_type"] = (
    "Estimated VIIRS footprint"
)


# ============================================================
# SAVE
# ============================================================

footprints.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("VIIRS FOOTPRINT DATASET CREATED")
print("=" * 70)

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(footprints):,}"
)

print("\nExample:")

print(
    footprints[
        [
            "latitude",
            "longitude",
            "scan",
            "track",
            "min_lat",
            "max_lat",
            "min_lon",
            "max_lon",
            "footprint_area_km2"
        ]
    ].head(5).to_string(index=False)
)

print("\nDone.")