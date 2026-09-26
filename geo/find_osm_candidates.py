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
    / "osm_fire_candidates.csv"
)


# ============================================================
# SELECT EVENT
# ============================================================

EVENT_INDEX = 0


# ============================================================
# LOAD FIRMS
# ============================================================

print("\nLoading VIIRS footprint data...")

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
# FOOTPRINT GEOMETRY
# ============================================================

footprint = box(
    event["min_lon"],
    event["min_lat"],
    event["max_lon"],
    event["max_lat"]
)


footprint_gdf = gpd.GeoDataFrame(
    [event],
    geometry=[footprint],
    crs="EPSG:4326"
)


# ============================================================
# LOAD OSM LANDUSE
# ============================================================

print("\nLoading OSM land-use layer...")

osm = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_landuse_a_free"
)

print(
    f"OSM polygons: {len(osm):,}"
)


# ============================================================
# CRS
# ============================================================

osm = osm.to_crs(
    "EPSG:4326"
)


# ============================================================
# FIND INTERSECTIONS
# ============================================================

print("\nSearching OSM polygons inside footprint...")

candidates = gpd.sjoin(
    osm,
    footprint_gdf[
        ["geometry"]
    ],
    how="inner",
    predicate="intersects"
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("OSM CANDIDATES")
print("=" * 70)

if len(candidates) == 0:

    print(
        "No OSM land-use polygon intersects "
        "this VIIRS footprint."
    )

else:

    print(
        f"Found {len(candidates)} candidate polygons."
    )

    print(
        candidates[
            [
                "osm_id",
                "fclass",
                "name"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

if len(candidates) > 0:

    output = candidates[
        [
            "osm_id",
            "fclass",
            "name",
            "geometry"
        ]
    ].copy()

    output.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )

    print(
        f"\nSaved candidates to:\n{OUTPUT_FILE}"
    )

print("\nDone.")