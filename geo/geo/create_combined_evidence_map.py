import rasterio
import numpy as np
import folium

from pathlib import Path
from pyproj import Transformer
from rasterio.windows import from_bounds
from branca.colormap import linear


# ============================================================
# CONFIG
# ============================================================

EVENT_ID = 84111

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


# ============================================================
# PROJECTED HOTSPOT
# ============================================================

with rasterio.open(B04_FILE) as src:

    crs = src.crs

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    hotspot_x, hotspot_y = transformer.transform(
        LON,
        LAT
    )

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

    window_transform = src.window_transform(window)


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
        out_shape=(
            b04.shape[0],
            b04.shape[1]
        ),
        resampling=rasterio.enums.Resampling.bilinear
    ).astype(np.float32)


# ============================================================
# SCALE REFLECTANCE
# ============================================================

b04 /= 10000.0
b08 /= 10000.0
b12 /= 10000.0


# ============================================================
# NDVI
# ============================================================

ndvi_denominator = b08 + b04

ndvi = np.divide(
    b08 - b04,
    ndvi_denominator,
    out=np.zeros_like(b08),
    where=ndvi_denominator != 0
)


# ============================================================
# NBR
# ============================================================

nbr_denominator = b08 + b12

nbr = np.divide(
    b08 - b12,
    nbr_denominator,
    out=np.zeros_like(b08),
    where=nbr_denominator != 0
)


# ============================================================
# RASTER EXTENT → LAT/LON
# ============================================================

height, width = ndvi.shape

corner_x = [
    min_x,
    max_x
]

corner_y = [
    min_y,
    max_y
]

to_latlon = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)

west, south = to_latlon.transform(
    min_x,
    min_y
)

east, north = to_latlon.transform(
    max_x,
    max_y
)

bounds = [
    [south, west],
    [north, east]
]


# ============================================================
# CREATE NDVI RGBA IMAGE
# ============================================================

ndvi_min = -0.2
ndvi_max = 0.5

normalized = (
    ndvi - ndvi_min
) / (
    ndvi_max - ndvi_min
)

normalized = np.clip(
    normalized,
    0,
    1
)

# Use matplotlib colormap only to create raster pixels
import matplotlib.pyplot as plt

cmap = plt.get_cmap("RdYlGn")

rgba = (
    cmap(normalized) * 255
).astype(np.uint8)

# Lower opacity
rgba[:, :, 3] = 130

# Invalid pixels transparent
invalid = ~np.isfinite(ndvi)

rgba[invalid, 3] = 0


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(
    location=[LAT, LON],
    zoom_start=15,
    control_scale=True
)


# ============================================================
# BASEMAP
# ============================================================

folium.TileLayer(
    "OpenStreetMap",
    name="OpenStreetMap"
).add_to(m)


# ============================================================
# NDVI RASTER OVERLAY
# ============================================================

folium.raster_layers.ImageOverlay(
    image=rgba,
    bounds=bounds,
    opacity=0.65,
    interactive=True,
    cross_origin=False,
    zindex=2,
    name="Sentinel-2 NDVI"
).add_to(m)


# ============================================================
# FIRMS HOTSPOT
# ============================================================

folium.Marker(
    location=[LAT, LON],
    tooltip=f"🔥 FIRMS Event {EVENT_ID}",
    popup=folium.Popup(
        f"""
        <b>FIRMS Thermal Hotspot</b><br><br>

        Event ID: {EVENT_ID}<br>
        Latitude: {LAT}<br>
        Longitude: {LON}<br><br>

        Source: NASA FIRMS<br>
        Type: Satellite thermal hotspot
        """,
        max_width=320
    ),
    icon=folium.Icon(
        color="red",
        icon="fire",
        prefix="fa"
    )
).add_to(m)


# ============================================================
# ESTIMATED VIIRS FOOTPRINT
# ============================================================

# Approximate ~375 m footprint.
# This is NOT the actual VIIRS pixel geometry.

footprint_half_lat = 0.00169
footprint_half_lon = 0.00180

footprint = [
    [
        LAT - footprint_half_lat,
        LON - footprint_half_lon
    ],
    [
        LAT - footprint_half_lat,
        LON + footprint_half_lon
    ],
    [
        LAT + footprint_half_lat,
        LON + footprint_half_lon
    ],
    [
        LAT + footprint_half_lat,
        LON - footprint_half_lon
    ]
]

folium.Polygon(
    locations=footprint,
    tooltip="Estimated VIIRS footprint",
    popup=folium.Popup(
        """
        <b>Estimated VIIRS footprint</b><br><br>

        Approximate satellite pixel-scale area.<br><br>

        ⚠️ This does not represent the exact
        fire boundary.
        """,
        max_width=320
    ),
    color="red",
    weight=3,
    fill=True,
    fill_opacity=0.12
).add_to(m)


# ============================================================
# HOTSPOT CENTER CIRCLE
# ============================================================

folium.Circle(
    location=[LAT, LON],
    radius=50,
    tooltip="FIRMS hotspot center",
    color="red",
    fill=False,
    weight=2
).add_to(m)


# ============================================================
# CENTER PIXEL VALUES
# ============================================================

center_row = ndvi.shape[0] // 2
center_col = ndvi.shape[1] // 2

center_ndvi = float(
    ndvi[center_row, center_col]
)

center_nbr = float(
    nbr[center_row, center_col]
)


# ============================================================
# NDVI LEGEND
# ============================================================

colormap = linear.RdYlGn_03.scale(
    ndvi_min,
    ndvi_max
)

colormap.caption = (
    "Sentinel-2 NDVI contextual layer"
)

colormap.add_to(m)


# ============================================================
# INFORMATION PANEL
# ============================================================

info_html = f"""
<div style="
position: fixed;
top: 20px;
right: 20px;
z-index: 9999;
background: white;
padding: 16px;
border: 2px solid #444;
border-radius: 10px;
font-size: 13px;
width: 290px;
box-shadow: 0 2px 10px rgba(0,0,0,0.2);
">

<div style="font-size:17px;">
<b>🔥 Event {EVENT_ID}</b>
</div>

<hr>

<b>FIRMS Hotspot</b><br>
Latitude: {LAT}<br>
Longitude: {LON}

<br><br>

<b>Sentinel-2 Evidence</b><br>
Center NDVI: {center_ndvi:.3f}<br>
Center NBR: {center_nbr:.3f}

<br><br>

<b>Evidence interpretation</b><br>
Sentinel-2 provides contextual
surface information around the
thermal hotspot.

<br><br>

<b>⚠️ Important</b><br>
The VIIRS footprint is estimated
and does not represent the exact
fire boundary.

</div>
"""

m.get_root().html.add_child(
    folium.Element(info_html)
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# FIT MAP TO AREA
# ============================================================

m.fit_bounds(bounds)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "event_84111_raster_evidence_map.html"
)

m.save(output_file)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("RASTER EVIDENCE MAP CREATED")
print("=" * 60)

print("Event ID:", EVENT_ID)
print("Hotspot:", LAT, LON)

print("Center NDVI:", center_ndvi)
print("Center NBR :", center_nbr)

print("\nRaster shape:", ndvi.shape)

print("\nSaved:")
print(output_file)