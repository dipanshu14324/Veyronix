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
SEARCH_RADIUS_DEG = 0.01


# ============================================================
# LOAD FIRMS
# ============================================================

print("\nLoading FIRMS...")

firms = pd.read_csv(FIRMS_FILE)

print(f"Total FIRMS hotspots: {len(firms):,}")


# ============================================================
# FILTER TO OSM COVERAGE
# ============================================================

firms_osm = firms[
    (firms["latitude"] >= 14.90) &
    (firms["latitude"] <= 24.71) &
    (firms["longitude"] >= 68.53) &
    (firms["longitude"] <= 80.73)
].copy()

print(
    f"Hotspots inside OSM coverage: "
    f"{len(firms_osm):,}"
)

print(
    f"Coverage of FIRMS: "
    f"{len(firms_osm) / len(firms) * 100:.2f}%"
)


# Take first 100 covered hotspots

sample = firms_osm.head(SAMPLE_SIZE).copy()

print(
    f"Testing {len(sample)} hotspots"
)


# ============================================================
# LOAD OSM BUILDINGS
# ============================================================

print("\nLoading OSM buildings...")

buildings = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_buildings_a_free"
)

print(
    f"Buildings loaded: {len(buildings):,}"
)

buildings = buildings.to_crs("EPSG:4326")

spatial_index = buildings.sindex


# ============================================================
# SEARCH BUILDINGS
# ============================================================

results = []

print("\nTesting hotspots...")


for i, row in sample.iterrows():

    point = Point(
        row["longitude"],
        row["latitude"]
    )

    search_area = point.buffer(
        SEARCH_RADIUS_DEG
    )

    possible_matches = list(
        spatial_index.intersection(
            search_area.bounds
        )
    )

    if possible_matches:

        nearby = buildings.iloc[possible_matches]

        nearby = nearby[
            nearby.geometry.intersects(
                search_area
            )
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


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 70)

print("BUILDING COVERAGE TEST V3")

print("=" * 70)

print(
    f"Hotspots tested: "
    f"{len(results_df)}"
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


print("\nTop hotspots:")

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