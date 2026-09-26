from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


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

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "osm_building_candidates.geojson"
)


# ============================================================
# SELECT HOTSPOT
# ============================================================

EVENT_INDEX = 0


# ============================================================
# LOAD FIRMS
# ============================================================

print("\nLoading VIIRS footprint...")

firms = pd.read_csv(FIRMS_FILE)

event = firms.iloc[EVENT_INDEX]

print("\nSelected hotspot:")
print(
    event[
        [
            "latitude",
            "longitude",
            "scan",
            "track",
            "footprint_area_km2"
        ]
    ]
)


# ============================================================
# CREATE FOOTPRINT
# ============================================================

footprint = box(
    event["min_lon"],
    event["min_lat"],
    event["max_lon"],
    event["max_lat"]
)


# ============================================================
# CREATE SEARCH AREA
# ============================================================

# We search slightly beyond the footprint because the
# satellite hotspot is not guaranteed to fall exactly
# on the building containing the fire.

SEARCH_BUFFER_DEGREES = 0.01

search_area = footprint.buffer(
    SEARCH_BUFFER_DEGREES
)


search_gdf = gpd.GeoDataFrame(
    geometry=[search_area],
    crs="EPSG:4326"
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
    f"Total OSM buildings: {len(buildings):,}"
)


# ============================================================
# CRS
# ============================================================

buildings = buildings.to_crs(
    "EPSG:4326"
)


# ============================================================
# SPATIAL SEARCH
# ============================================================

print("\nSearching nearby buildings...")

candidates = gpd.sjoin(
    buildings,
    search_gdf,
    how="inner",
    predicate="intersects"
)


# ============================================================
# CALCULATE DISTANCE
# ============================================================

# Project to metres for accurate distance calculation.

candidates = candidates.to_crs(
    "EPSG:32644"
)

hotspot_gdf = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(
        [event.geometry]
        if "geometry" in event
        else []
    ),
    crs="EPSG:4326"
)


# Create hotspot point directly.

from shapely.geometry import Point

hotspot_point = gpd.GeoSeries(
    [
        Point(
            event["longitude"],
            event["latitude"]
        )
    ],
    crs="EPSG:4326"
).to_crs("EPSG:32644").iloc[0]


candidates["distance_to_hotspot_m"] = (
    candidates.geometry
    .distance(hotspot_point)
)


# ============================================================
# DISTANCE RANK
# ============================================================

candidates = candidates.sort_values(
    "distance_to_hotspot_m"
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("NEARBY OSM BUILDINGS")
print("=" * 70)

print(
    f"Buildings found: {len(candidates):,}"
)


if len(candidates) > 0:

    display_columns = [
        col
        for col in [
            "osm_id",
            "code",
            "fclass",
            "name",
            "distance_to_hotspot_m"
        ]
        if col in candidates.columns
    ]

    print(
        candidates[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

if len(candidates) > 0:

    candidates = candidates.to_crs(
        "EPSG:4326"
    )

    candidates.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

else:

    print(
        "\nNo buildings found in the search area."
    )


print("\nDone.")