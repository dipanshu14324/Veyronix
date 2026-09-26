from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR = Path(__file__).resolve().parent.parent

GPKG_FILE = (
    BASE_DIR
    / "data"
    / "raw_data"
    / "western-zone-260923-free.gpkg"
    / "western-zone.gpkg"
)

FIRMS_FILE = (
    BASE_DIR
    / "data"
    / "processed_data"
    / "phase1_viirs_footprints.csv"
)

print("\nLoading buildings...")

buildings = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_buildings_a_free"
)

print("CRS:", buildings.crs)
print("Buildings:", len(buildings))

print("\nFirst building:")
print(buildings.iloc[0])

print("\nBuilding bounds:")
print(buildings.total_bounds)


print("\nLoading FIRMS...")

firms = pd.read_csv(FIRMS_FILE)

# Use the first hotspot that is inside OSM coverage
hotspot = firms[
    (firms["latitude"] >= 14.90) &
    (firms["latitude"] <= 24.71) &
    (firms["longitude"] >= 68.53) &
    (firms["longitude"] <= 80.73)
].iloc[0]

lat = hotspot["latitude"]
lon = hotspot["longitude"]

print("\nTest hotspot:")
print("Latitude :", lat)
print("Longitude:", lon)


# ------------------------------------------------------------
# CREATE POINT
# ------------------------------------------------------------

point = Point(lon, lat)

print("\nPoint:")
print(point)

print("\nPoint CRS: EPSG:4326")


# ------------------------------------------------------------
# CHECK CRS
# ------------------------------------------------------------

print("\nConverting buildings to EPSG:4326...")

buildings = buildings.to_crs("EPSG:4326")

print("New CRS:", buildings.crs)


# ------------------------------------------------------------
# MANUAL BOUNDING BOX
# ------------------------------------------------------------

buffer = 0.1

min_lon = lon - buffer
max_lon = lon + buffer

min_lat = lat - buffer
max_lat = lat + buffer

print("\nSearch bounding box:")

print("Longitude:", min_lon, "to", max_lon)
print("Latitude :", min_lat, "to", max_lat)


# ------------------------------------------------------------
# CHECK USING GEOMETRY BOUNDS
# ------------------------------------------------------------

candidate = buildings[
    (buildings.geometry.bounds["minx"] <= max_lon) &
    (buildings.geometry.bounds["maxx"] >= min_lon) &
    (buildings.geometry.bounds["miny"] <= max_lat) &
    (buildings.geometry.bounds["maxy"] >= min_lat)
]

print("\nBuildings whose bounding boxes overlap:")
print(len(candidate))


# ------------------------------------------------------------
# DISTANCE APPROXIMATION
# ------------------------------------------------------------

if len(candidate) > 0:

    candidate = candidate.copy()

    candidate["distance"] = candidate.geometry.centroid.distance(point)

    print("\nNearest candidate buildings:")

    print(
        candidate[
            ["osm_id", "fclass", "name", "distance"]
        ]
        .sort_values("distance")
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "\nNO BUILDING GEOMETRY EXISTS "
        "WITHIN 0.1 DEGREE OF THE HOTSPOT."
    )


print("\nDone.")
