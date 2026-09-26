from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

FIRMS_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "phase1_viirs_footprints.csv"
)

GPKG_FILE = (
    BASE_DIR
    / "data"
    / "raw_data"
    / "western-zone-260923-free.gpkg"
    / "western-zone.gpkg"
)


print("\nLoading FIRMS...")

firms = pd.read_csv(FIRMS_FILE)

print("FIRMS bounds:")
print(
    "Latitude:",
    firms["latitude"].min(),
    "to",
    firms["latitude"].max()
)

print(
    "Longitude:",
    firms["longitude"].min(),
    "to",
    firms["longitude"].max()
)


print("\nLoading OSM buildings...")

buildings = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_buildings_a_free"
)

print(f"Buildings: {len(buildings):,}")

buildings = buildings.to_crs("EPSG:4326")

print("\nOSM building bounds:")

minx, miny, maxx, maxy = buildings.total_bounds

print("Longitude:", minx, "to", maxx)
print("Latitude :", miny, "to", maxy)


print("\n" + "=" * 70)
print("COVERAGE CHECK")
print("=" * 70)

print(
    f"FIRMS latitude range : "
    f"{firms['latitude'].min():.2f} → "
    f"{firms['latitude'].max():.2f}"
)

print(
    f"FIRMS longitude range: "
    f"{firms['longitude'].min():.2f} → "
    f"{firms['longitude'].max():.2f}"
)

print(
    f"OSM latitude range   : "
    f"{miny:.2f} → {maxy:.2f}"
)

print(
    f"OSM longitude range  : "
    f"{minx:.2f} → {maxx:.2f}"
)

print("\nDone.")