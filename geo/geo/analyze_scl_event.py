import rasterio
import numpy as np
from pathlib import Path
from pyproj import Transformer

# ============================================================
# CONFIG
# ============================================================

LAT = 18.6813
LON = 73.0320

BASE_DIR = Path.cwd()
SENTINEL_DIR = BASE_DIR / "data" / "sentinel" / "event_84111"

SCL_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_SCL_20m.jp2"

B04_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_B04_10m.jp2"

# ============================================================
# SCL CLASS NAMES
# ============================================================

SCL_CLASSES = {
    0: "No Data",
    1: "Saturated / Defective",
    2: "Dark Area / Shadow",
    3: "Cloud Shadow",
    4: "Vegetation",
    5: "Not Vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud - Medium Probability",
    9: "Cloud - High Probability",
    10: "Thin Cirrus",
    11: "Snow / Ice",
}

# ============================================================
# READ B04 GRID
# ============================================================

with rasterio.open(B04_FILE) as b04:

    crs = b04.crs
    transform = b04.transform

    width = b04.width
    height = b04.height

print("B04 CRS:", crs)
print("B04 size:", width, "x", height)

# ============================================================
# CONVERT LAT/LON → UTM
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    crs,
    always_xy=True
)

x, y = transformer.transform(LON, LAT)

print("\nHotspot coordinates:")
print("Latitude :", LAT)
print("Longitude:", LON)
print("Projected X:", x)
print("Projected Y:", y)

# ============================================================
# READ SCL
# ============================================================

with rasterio.open(SCL_FILE) as scl:

    # Convert hotspot coordinate to SCL pixel
    row, col = scl.index(x, y)

    print("\nSCL raster:")
    print("Size:", scl.width, "x", scl.height)
    print("CRS :", scl.crs)

    # Center pixel
    center_value = int(
        scl.read(1)[row, col]
    )

    print("\nHotspot center SCL:")
    print("Class:", center_value)
    print("Meaning:", SCL_CLASSES.get(
        center_value,
        "Unknown"
    ))

    # ========================================================
    # 1 KM × 1 KM WINDOW
    # ========================================================

    buffer_pixels = 25   # approximately 500 m at 20 m resolution

    r1 = max(0, row - buffer_pixels)
    r2 = min(scl.height, row + buffer_pixels)

    c1 = max(0, col - buffer_pixels)
    c2 = min(scl.width, col + buffer_pixels)

    crop = scl.read(
        1,
        window=((r1, r2), (c1, c2))
    )

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

values, counts = np.unique(
    crop,
    return_counts=True
)

total = crop.size

print("\n" + "=" * 60)
print("SCL CLASS DISTRIBUTION — ~1 KM × 1 KM")
print("=" * 60)

for value, count in zip(values, counts):

    value = int(value)

    percentage = (
        count / total * 100
    )

    name = SCL_CLASSES.get(
        value,
        "Unknown"
    )

    print(
        f"{value:2d} | "
        f"{name:<30} | "
        f"{count:5d} pixels | "
        f"{percentage:6.2f}%"
    )

# ============================================================
# QUALITY CHECK
# ============================================================

BAD_CLASSES = {
    0, 1, 3, 8, 9, 10, 11
}

bad_pixels = sum(
    count
    for value, count in zip(values, counts)
    if int(value) in BAD_CLASSES
)

bad_percentage = (
    bad_pixels / total * 100
)

print("\n" + "=" * 60)
print("IMAGE QUALITY")
print("=" * 60)

print(
    f"Potentially unusable/cloud pixels: "
    f"{bad_percentage:.2f}%"
)

if center_value in BAD_CLASSES:

    print(
        "\n⚠️ HOTSPOT CENTER HAS A QUALITY ISSUE:"
    )

    print(
        SCL_CLASSES.get(
            center_value,
            "Unknown"
        )
    )

else:

    print(
        "\n✅ HOTSPOT CENTER IS NOT FLAGGED "
        "AS CLOUD/SHADOW/INVALID."
    )

# ============================================================
# FINAL INTERPRETATION
# ============================================================

if center_value in {8, 9, 10}:

    print(
        "\n❌ Sentinel optical evidence should "
        "NOT be trusted at the hotspot center."
    )

elif center_value in {0, 1, 3}:

    print(
        "\n⚠️ Center pixel has invalid/shadow "
        "conditions."
    )

elif center_value in {4, 5}:

    print(
        "\n✅ Surface pixel is usable for "
        "NDVI/NBR contextual analysis."
    )

else:

    print(
        "\nℹ️ Center pixel has another valid "
        "surface classification."
    )