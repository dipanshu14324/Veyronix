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

SAMPLE_SIZE = 500


print("\nLoading FIRMS...")

firms = pd.read_csv(FIRMS_FILE)

sample = firms.head(SAMPLE_SIZE).copy()

print(f"Testing {len(sample)} hotspots")


print("\nLoading OSM land-use...")

landuse = gpd.read_file(
    GPKG_FILE,
    layer="gis_osm_landuse_a_free"
)

print(f"Land-use polygons: {len(landuse):,}")

landuse = landuse.to_crs("EPSG:4326")

print("\nLand-use classes:")

print(
    landuse["fclass"]
    .value_counts()
    .head(20)
)


# Spatial index
spatial_index = landuse.sindex


results = []


print("\nTesting hotspots...")


for i, row in sample.iterrows():

    point = Point(
        row["longitude"],
        row["latitude"]
    )

    # Small search area around hotspot
    search_area = point.buffer(0.005)

    possible = list(
        spatial_index.intersection(
            search_area.bounds
        )
    )

    if possible:

        nearby = landuse.iloc[possible]

        nearby = nearby[
            nearby.geometry.intersects(search_area)
        ]

    else:

        nearby = landuse.iloc[[]]


    classes = nearby["fclass"].tolist()

    results.append({
        "index": i,
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "landuse_count": len(nearby),
        "landuse_classes": ",".join(
            sorted(set(classes))
        )
    })


results_df = pd.DataFrame(results)


print("\n" + "=" * 70)
print("LAND-USE COVERAGE TEST")
print("=" * 70)

print(
    f"Hotspots tested: {len(results_df)}"
)

print(
    f"With land-use: "
    f"{(results_df['landuse_count'] > 0).sum()}"
)

print(
    f"Without land-use: "
    f"{(results_df['landuse_count'] == 0).sum()}"
)

print(
    f"Coverage: "
    f"{(results_df['landuse_count'] > 0).mean() * 100:.2f}%"
)


print("\nMost common land-use classes near hotspots:")

all_classes = []

for value in results_df["landuse_classes"]:

    if value:

        all_classes.extend(
            value.split(",")
        )

if all_classes:

    print(
        pd.Series(all_classes)
        .value_counts()
        .head(20)
    )


print("\nSample results:")

print(
    results_df
    .head(20)
    .to_string(index=False)
)


print("\nDone.")