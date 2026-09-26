import geopandas as gpd

GPKG_FILE = (
    r"data/raw_data/"
    r"western-zone-260923-free.gpkg/"
    r"western-zone.gpkg"
)

layers = gpd.list_layers(GPKG_FILE)

print("\nOSM LAYERS")
print("=" * 80)

for _, row in layers.iterrows():
    print(
        f"{row['name']:<40} "
        f"{row['geometry_type']}"
    )