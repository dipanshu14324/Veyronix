from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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

SAMPLE_SIZE = 100

# Approximately 1 km in latitude
SEARCH_RADIUS_DEG = 0.01


print("\nLoading FIRMS footprints...")

firms = pd.read_csv(FIRMS_FILE)

sample = firms.head(SAMPLE_SIZE).copy()

print(f"Testing {len(sample)} hotspots")


print("\nLoading OSM buildings...")

buildings = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_buildings_a_free"
)

print(f"Buildings loaded: {len(buildings):,}")

# Force geographic CRS
buildings = buildings.to_crs("EPSG:4326")

spatial_index = buildings.sindex


results = []

print("\nTesting hotspots...")


for i, row in sample.iterrows():

    point = Point(
        row["longitude"],
        row["latitude"]
    )

    search_area = point.buffer(SEARCH_RADIUS_DEG)

    possible_matches = list(
        spatial_index.intersection(
            search_area.bounds
        )
    )

    if possible_matches:

        nearby = buildings.iloc[possible_matches]

        nearby = nearby[
            nearby.geometry.intersects(search_area)
        ]

        building_count = len(nearby)

    else:

        building_count = 0


    results.append({
        "index": i,
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "scan": row["scan"],
        "track": row["track"],
        "building_count": building_count
    })


results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("BUILDING COVERAGE TEST V2")
print("=" * 70)

print(f"Hotspots tested: {len(results_df)}")

print(
    f"With buildings: "
    f"{(results_df['building_count'] > 0).sum()}"
)

print(
    f"Without buildings: "
    f"{(results_df['building_count'] == 0).sum()}"
)

print(
    f"Coverage: "
    f"{(results_df['building_count'] > 0).mean() * 100:.2f}%"
)


print("\nBuilding count statistics:")

print(
    results_df["building_count"].describe()
)


print("\nTop hotspots by building count:")

print(
    results_df
    .sort_values(
        "building_count",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\nDone.")