import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from pathlib import Path

# ============================================================
# EVENT
# ============================================================

LAT = 18.6813
LON = 73.0320

# Analyze approximately 1 km × 1 km around hotspot
BUFFER_M = 500

OUTPUT_DIR = Path("data/sentinel/event_84111")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

B04_FILE = OUTPUT_DIR / "T43QBA_20250131T054121_B04_10m.jp2"
B08_FILE = OUTPUT_DIR / "T43QBA_20250131T054121_B08_10m.jp2"
B12_FILE = OUTPUT_DIR / "T43QBA_20250131T054121_B12_20m.jp2"


# ============================================================
# OPEN B04
# ============================================================

with rasterio.open(B04_FILE) as b04:

    # Convert geographic coordinate to UTM
    from rasterio.warp import transform

    x, y = transform(
        "EPSG:4326",
        b04.crs,
        [LON],
        [LAT]
    )

    x = x[0]
    y = y[0]

    print("Hotspot projected coordinate:")
    print("X:", x)
    print("Y:", y)

    # Create 1 km × 1 km window
    min_x = x - BUFFER_M
    max_x = x + BUFFER_M
    min_y = y - BUFFER_M
    max_y = y + BUFFER_M

    window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=b04.transform
    )

    # Round window to integer pixels
    window = window.round_offsets().round_lengths()

    b04_data = b04.read(
        1,
        window=window
    ).astype("float32")

    window_transform = b04.window_transform(window)

    print("\nB04 crop:")
    print("Shape:", b04_data.shape)


# ============================================================
# OPEN B08
# ============================================================

with rasterio.open(B08_FILE) as b08:

    b08_data = b08.read(
        1,
        window=window
    ).astype("float32")

    print("\nB08 crop:")
    print("Shape:", b08_data.shape)


# ============================================================
# OPEN B12
# ============================================================

with rasterio.open(B12_FILE) as b12:

    # Allocate B12 at B04 resolution
    b12_data = np.empty_like(b04_data)

    reproject(
        source=rasterio.band(b12, 1),
        destination=b12_data,
        src_transform=b12.transform,
        src_crs=b12.crs,
        dst_transform=window_transform,
        dst_crs=b04.crs,
        resampling=Resampling.bilinear
    )

    print("\nB12 resampled:")
    print("Shape:", b12_data.shape)


# ============================================================
# SCALE SENTINEL REFLECTANCE
# ============================================================

# Sentinel-2 L2A JP2 values are normally scaled by 10000.

b04 = b04_data / 10000.0
b08 = b08_data / 10000.0
b12 = b12_data / 10000.0


# ============================================================
# NDVI
# ============================================================

ndvi_denominator = b08 + b04

ndvi = np.divide(
    b08 - b04,
    ndvi_denominator,
    out=np.full_like(b08, np.nan),
    where=ndvi_denominator != 0
)


# ============================================================
# NBR
# ============================================================

nbr_denominator = b08 + b12

nbr = np.divide(
    b08 - b12,
    nbr_denominator,
    out=np.full_like(b08, np.nan),
    where=nbr_denominator != 0
)


# ============================================================
# BASIC STATISTICS
# ============================================================

def stats(name, data):

    valid = data[np.isfinite(data)]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Valid pixels:", len(valid))

    if len(valid) == 0:
        print("No valid pixels.")
        return

    print("Min :", np.nanmin(valid))
    print("Mean:", np.nanmean(valid))
    print("Max :", np.nanmax(valid))
    print("Median:", np.nanmedian(valid))


stats("B04 Reflectance", b04)
stats("B08 Reflectance", b08)
stats("B12 Reflectance", b12)

stats("NDVI", ndvi)
stats("NBR", nbr)


# ============================================================
# HOTSPOT CENTER PIXEL
# ============================================================

hotspot_row, hotspot_col = rasterio.transform.rowcol(
    window_transform,
    x,
    y
)

print("\n" + "=" * 60)
print("HOTSPOT CENTER PIXEL")
print("=" * 60)

print("Row:", hotspot_row)
print("Column:", hotspot_col)

if (
    0 <= hotspot_row < ndvi.shape[0]
    and
    0 <= hotspot_col < ndvi.shape[1]
):

    print("B04:", b04[hotspot_row, hotspot_col])
    print("B08:", b08[hotspot_row, hotspot_col])
    print("B12:", b12[hotspot_row, hotspot_col])
    print("NDVI:", ndvi[hotspot_row, hotspot_col])
    print("NBR:", nbr[hotspot_row, hotspot_col])


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    OUTPUT_DIR / "ndvi_event_84111.npy",
    ndvi
)

np.save(
    OUTPUT_DIR / "nbr_event_84111.npy",
    nbr
)

np.save(
    OUTPUT_DIR / "b04_event_84111.npy",
    b04
)

np.save(
    OUTPUT_DIR / "b08_event_84111.npy",
    b08
)

np.save(
    OUTPUT_DIR / "b12_event_84111.npy",
    b12
)

print("\nSaved NDVI/NBR arrays.")

print("\nAnalysis complete.")