from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 100

# Search radius around hotspot
SEARCH_RADIUS_M = 1000


# ============================================================
# LOAD FIRMS
# ============================================================

print("\nLoading FIRMS footprints...")

firms = pd.read_csv(FIRMS_FILE)

sample = firms.head(SAMPLE_SIZE).copy()

print(
    f"Testing {len(sample)} hotspots"
)


# ============================================================
# LOAD BUILDINGS
# ============================================================

print("\nLoading OSM buildings...")

buildings = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_buildings_a_free"
)

print(
    f"Buildings loaded: {len(buildings):,}"
)


# ============================================================
# PROJECT TO METRES
# ============================================================

buildings = buildings.to_crs(
    "EPSG:32644"
)


# ============================================================
# SPATIAL INDEX
# ============================================================

spatial_index = buildings.sindex


# ============================================================
# TEST HOTSPOTS
# ============================================================

results = []

print("\nTesting hotspots...")

for i, row in sample.iterrows():

    point = Point(
        row["longitude"],
        row["latitude"]
    )

    point_gdf = gpd.GeoSeries(
        [point],
        crs="EPSG:4326"
    ).to_crs("EPSG:32644")

    point_m = point_gdf.iloc[0]

    search_area = point_m.buffer(
        SEARCH_RADIUS_M
    )

    possible_matches = list(
        spatial_index.intersection(
            search_area.bounds
        )
    )

    if possible_matches:

        nearby = buildings.iloc[
            possible_matches
        ]

        nearby = nearby[
            nearby.geometry.intersects(
                search_area
            )
        ]

        building_count = len(nearby)

    else:

        building_count = 0


    results.append(
        {
            "index": i,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "scan": row["scan"],
            "track": row["track"],
            "building_count": building_count
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("BUILDING COVERAGE TEST")
print("=" * 70)

print(
    f"Hotspots tested: {len(results_df)}"
)

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
    results_df[
        "building_count"
    ].describe()
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