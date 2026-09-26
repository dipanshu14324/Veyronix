import rasterio
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pyproj import Transformer
from rasterio.windows import from_bounds


# ============================================================
# CONFIG
# ============================================================

LAT = 18.6813
LON = 73.0320

BASE_DIR = Path.cwd()

SENTINEL_DIR = (
    BASE_DIR
    / "data"
    / "sentinel"
    / "event_84111"
)

OUTPUT_DIR = (
    SENTINEL_DIR
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

B04_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_B04_10m.jp2"
B08_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_B08_10m.jp2"
B12_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_B12_20m.jp2"
SCL_FILE = SENTINEL_DIR / "T43QBA_20250131T054121_SCL_20m.jp2"


# ============================================================
# LOAD B04 GRID
# ============================================================

with rasterio.open(B04_FILE) as src:

    crs = src.crs
    transform = src.transform

    hotspot_x, hotspot_y = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    ).transform(LON, LAT)

    # 500 m around hotspot
    min_x = hotspot_x - 500
    max_x = hotspot_x + 500
    min_y = hotspot_y - 500
    max_y = hotspot_y + 500

    window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=src.transform
    )

    b04 = src.read(
        1,
        window=window
    ).astype(np.float32)

    b04_transform = src.window_transform(window)


# ============================================================
# LOAD B08
# ============================================================

with rasterio.open(B08_FILE) as src:

    b08 = src.read(
        1,
        window=window
    ).astype(np.float32)


# ============================================================
# LOAD B12
# ============================================================

with rasterio.open(B12_FILE) as src:

    b12_window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=src.transform
    )

    b12 = src.read(
    1,
    window=b12_window,
    out_shape=(b04.shape[0], b04.shape[1]),
    resampling=rasterio.enums.Resampling.bilinear
).astype(np.float32)

# ============================================================
# LOAD SCL
# ============================================================

with rasterio.open(SCL_FILE) as src:

    scl_window = from_bounds(
        min_x,
        min_y,
        max_x,
        max_y,
        transform=src.transform
    )

    scl = src.read(
        1,
        window=scl_window,
        out_shape=(b04.shape[0], b04.shape[1]),
        resampling=rasterio.enums.Resampling.nearest
    )


# ============================================================
# SCALE REFLECTANCE
# ============================================================

b04 = b04 / 10000.0
b08 = b08 / 10000.0
b12 = b12 / 10000.0


# ============================================================
# CALCULATE NDVI
# ============================================================

ndvi_denominator = b08 + b04

ndvi = np.divide(
    b08 - b04,
    ndvi_denominator,
    out=np.zeros_like(b08),
    where=ndvi_denominator != 0
)


# ============================================================
# CALCULATE NBR
# ============================================================

nbr_denominator = b08 + b12

nbr = np.divide(
    b08 - b12,
    nbr_denominator,
    out=np.zeros_like(b08),
    where=nbr_denominator != 0
)


# ============================================================
# HOTSPOT POSITION IN IMAGE
# ============================================================

hotspot_col = int(
    (hotspot_x - min_x)
    / (max_x - min_x)
    * b04.shape[1]
)

hotspot_row = int(
    (max_y - hotspot_y)
    / (max_y - min_y)
    * b04.shape[0]
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("=" * 60)
print("SENTINEL-2 EVIDENCE ANALYSIS")
print("=" * 60)

print(f"Event ID     : 84111")
print(f"Latitude     : {LAT}")
print(f"Longitude    : {LON}")

print("\nArray dimensions:")
print("B04 :", b04.shape)
print("B08 :", b08.shape)
print("B12 :", b12.shape)
print("SCL :", scl.shape)

print("\nNDVI:")
print("Min   :", np.nanmin(ndvi))
print("Mean  :", np.nanmean(ndvi))
print("Max   :", np.nanmax(ndvi))

print("\nNBR:")
print("Min   :", np.nanmin(nbr))
print("Mean  :", np.nanmean(nbr))
print("Max   :", np.nanmax(nbr))

print("\nHotspot image position:")
print("Row:", hotspot_row)
print("Col:", hotspot_col)


# ============================================================
# CREATE FIGURES
# ============================================================

figures = []


# ============================================================
# 1. B04 — RED BAND
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

ax.imshow(
    b04,
    cmap="gray"
)

ax.scatter(
    hotspot_col,
    hotspot_row,
    marker="x",
    s=120,
    linewidths=3
)

ax.set_title(
    "Sentinel-2 B04 — Red Band\nFIRMS Event 84111"
)

ax.set_xlabel("Pixel")
ax.set_ylabel("Pixel")

fig.tight_layout()

output = OUTPUT_DIR / "event_84111_B04.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved:", output)


# ============================================================
# 2. NDVI
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    ndvi,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)

ax.scatter(
    hotspot_col,
    hotspot_row,
    marker="x",
    s=120,
    linewidths=3
)

ax.set_title(
    "Sentinel-2 NDVI\nFIRMS Event 84111"
)

ax.set_xlabel("Pixel")
ax.set_ylabel("Pixel")

fig.colorbar(
    im,
    ax=ax,
    label="NDVI"
)

fig.tight_layout()

output = OUTPUT_DIR / "event_84111_NDVI.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", output)


# ============================================================
# 3. NBR
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    nbr,
    cmap="RdYlBu_r",
    vmin=-1,
    vmax=1
)

ax.scatter(
    hotspot_col,
    hotspot_row,
    marker="x",
    s=120,
    linewidths=3
)

ax.set_title(
    "Sentinel-2 NBR\nFIRMS Event 84111"
)

ax.set_xlabel("Pixel")
ax.set_ylabel("Pixel")

fig.colorbar(
    im,
    ax=ax,
    label="NBR"
)

fig.tight_layout()

output = OUTPUT_DIR / "event_84111_NBR.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", output)


# ============================================================
# 4. SCL
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

im = ax.imshow(
    scl,
    interpolation="nearest"
)

ax.scatter(
    hotspot_col,
    hotspot_row,
    marker="x",
    s=120,
    linewidths=3
)

ax.set_title(
    "Sentinel-2 Scene Classification\nFIRMS Event 84111"
)

ax.set_xlabel("Pixel")
ax.set_ylabel("Pixel")

fig.colorbar(
    im,
    ax=ax,
    label="SCL Class"
)

fig.tight_layout()

output = OUTPUT_DIR / "event_84111_SCL.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", output)


# ============================================================
# 5. RGB-STYLE VIEW
# ============================================================

# Sentinel-2 natural color:
# Red   = B04
# Green = B03
# Blue  = B02
#
# We don't currently have B02/B03 downloaded.
# Therefore use B08/B04/B12 as a false-color composite.

rgb = np.dstack([
    b12,
    b08,
    b04
])

# Robust percentile stretch

low = np.nanpercentile(
    rgb,
    2
)

high = np.nanpercentile(
    rgb,
    98
)

rgb_display = np.clip(
    (rgb - low) / (high - low),
    0,
    1
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

ax.imshow(
    rgb_display
)

ax.scatter(
    hotspot_col,
    hotspot_row,
    marker="x",
    s=150,
    linewidths=3
)

ax.set_title(
    "Sentinel-2 False Color Composite\nB12 / B08 / B04 — Event 84111"
)

ax.set_xlabel("Pixel")
ax.set_ylabel("Pixel")

fig.tight_layout()

output = OUTPUT_DIR / "event_84111_false_color.png"

fig.savefig(
    output,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved:", output)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print("\nOutput directory:")
print(OUTPUT_DIR)