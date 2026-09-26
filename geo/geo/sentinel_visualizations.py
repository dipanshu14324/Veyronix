from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import rowcol


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
SENTINEL_DIR = BASE_DIR / "data" / "sentinel"


# ============================================================
# FIND SENTINEL BAND
# ============================================================

def find_band_file(event_id, band_name):
    """
    Find a Sentinel band inside:
    data/sentinel/event_{event_id}/
    """

    event_dir = SENTINEL_DIR / f"event_{event_id}"

    if not event_dir.exists():
        raise FileNotFoundError(
            f"Sentinel directory not found: {event_dir}"
        )

    matches = list(event_dir.glob(f"*_{band_name}.jp2"))

    if not matches:
        raise FileNotFoundError(
            f"{band_name} file not found for event {event_id}"
        )

    return matches[0]


# ============================================================
# LOAD BAND WINDOW
# ============================================================

def load_window(dataset, latitude, longitude, buffer_m=500):
    """
    Read approximately a 1 km x 1 km window
    around the FIRMS hotspot.
    """

    # Convert geographic coordinates to raster CRS
    from rasterio.warp import transform

    x, y = transform(
        "EPSG:4326",
        dataset.crs,
        [longitude],
        [latitude]
    )

    x = x[0]
    y = y[0]

    half_size = buffer_m

    left = x - half_size
    right = x + half_size
    bottom = y - half_size
    top = y + half_size

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        dataset.transform
    )

    data = dataset.read(
        1,
        window=window
    ).astype(np.float32)

    transform_window = dataset.window_transform(window)

    return data, transform_window


# ============================================================
# RESAMPLE BAND
# ============================================================

def resample_to_reference(
    source_dataset,
    reference_dataset,
    latitude,
    longitude,
    buffer_m=500,
    resampling=Resampling.bilinear
):

    from rasterio.warp import transform

    x, y = transform(
        "EPSG:4326",
        reference_dataset.crs,
        [longitude],
        [latitude]
    )

    x = x[0]
    y = y[0]

    half_size = buffer_m

    left = x - half_size
    right = x + half_size
    bottom = y - half_size
    top = y + half_size

    ref_window = from_bounds(
        left,
        bottom,
        right,
        top,
        reference_dataset.transform
    )

    ref_transform = reference_dataset.window_transform(
        ref_window
    )

    ref_height = int(ref_window.height)
    ref_width = int(ref_window.width)

    destination = np.empty(
        (ref_height, ref_width),
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(source_dataset, 1),
        destination=destination,
        src_transform=source_dataset.transform,
        src_crs=source_dataset.crs,
        dst_transform=ref_transform,
        dst_crs=reference_dataset.crs,
        resampling=resampling
    )

    return destination, ref_transform


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    array,
    output_path,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None
):

    plt.figure(figsize=(7, 6))

    masked = np.ma.masked_invalid(array)

    plt.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# GENERATE SENTINEL VISUALIZATIONS
# ============================================================

def generate_sentinel_visualizations(
    event_id,
    latitude,
    longitude,
    buffer_m=500
):

    event_dir = SENTINEL_DIR / f"event_{event_id}"

    if not event_dir.exists():
        raise FileNotFoundError(
            f"Sentinel cache not found for event {event_id}"
        )

    output_dir = event_dir / "visualizations"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    b04_file = find_band_file(event_id, "B04_10m")
    b08_file = find_band_file(event_id, "B08_10m")
    b12_file = find_band_file(event_id, "B12_20m")
    scl_file = find_band_file(event_id, "SCL_20m")

    # --------------------------------------------------------
    # OPEN RASTERS
    # --------------------------------------------------------

    with rasterio.open(b04_file) as b04_ds, \
         rasterio.open(b08_file) as b08_ds, \
         rasterio.open(b12_file) as b12_ds, \
         rasterio.open(scl_file) as scl_ds:

        # ----------------------------------------------------
        # B04
        # ----------------------------------------------------

        b04, ref_transform = load_window(
            b04_ds,
            latitude,
            longitude,
            buffer_m
        )

        # ----------------------------------------------------
        # B08
        # ----------------------------------------------------

        b08, _ = load_window(
            b08_ds,
            latitude,
            longitude,
            buffer_m
        )

        # ----------------------------------------------------
        # B12 -> B04 GRID
        # ----------------------------------------------------

        b12, _ = resample_to_reference(
            b12_ds,
            b04_ds,
            latitude,
            longitude,
            buffer_m,
            Resampling.bilinear
        )

        # ----------------------------------------------------
        # SCL -> B04 GRID
        # ----------------------------------------------------

        scl, _ = resample_to_reference(
            scl_ds,
            b04_ds,
            latitude,
            longitude,
            buffer_m,
            Resampling.nearest
        )

    # --------------------------------------------------------
    # REFLECTANCE
    # --------------------------------------------------------

    b04 = b04 / 10000.0
    b08 = b08 / 10000.0
    b12 = b12 / 10000.0

    # --------------------------------------------------------
    # SAFE DIVISION
    # --------------------------------------------------------

    denominator_ndvi = b08 + b04
    denominator_nbr = b08 + b12

    ndvi = np.divide(
        b08 - b04,
        denominator_ndvi,
        out=np.full_like(b08, np.nan),
        where=denominator_ndvi != 0
    )

    nbr = np.divide(
        b08 - b12,
        denominator_nbr,
        out=np.full_like(b08, np.nan),
        where=denominator_nbr != 0
    )

    # --------------------------------------------------------
    # FALSE COLOR
    #
    # Simple SWIR-NIR-RED visualization:
    # R = B12
    # G = B08
    # B = B04
    # --------------------------------------------------------

    false_color = np.stack(
        [
            b12,
            b08,
            b04
        ],
        axis=-1
    )

    # Percentile stretch
    for channel in range(3):

        channel_data = false_color[:, :, channel]

        low = np.nanpercentile(
            channel_data,
            2
        )

        high = np.nanpercentile(
            channel_data,
            98
        )

        false_color[:, :, channel] = np.clip(
            (channel_data - low) /
            (high - low + 1e-8),
            0,
            1
        )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    ndvi_path = (
        output_dir /
        f"event_{event_id}_NDVI.png"
    )

    nbr_path = (
        output_dir /
        f"event_{event_id}_NBR.png"
    )

    scl_path = (
        output_dir /
        f"event_{event_id}_SCL.png"
    )

    false_color_path = (
        output_dir /
        f"event_{event_id}_false_color.png"
    )

    # --------------------------------------------------------
    # SAVE NDVI
    # --------------------------------------------------------

    save_image(
        ndvi,
        ndvi_path,
        f"Event {event_id} — NDVI",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    # --------------------------------------------------------
    # SAVE NBR
    # --------------------------------------------------------

    save_image(
        nbr,
        nbr_path,
        f"Event {event_id} — NBR",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )

    # --------------------------------------------------------
    # SAVE SCL
    # --------------------------------------------------------

    save_image(
        scl,
        scl_path,
        f"Event {event_id} — Sentinel Classification",
        cmap="tab10"
    )

    # --------------------------------------------------------
    # SAVE FALSE COLOR
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.imshow(false_color)

    plt.title(
        f"Event {event_id} — Sentinel-2 False Color"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        false_color_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # RETURN PATHS
    # --------------------------------------------------------

    return {
        "event_id": event_id,
        "visualizations": {
            "ndvi": str(ndvi_path),
            "nbr": str(nbr_path),
            "scl": str(scl_path),
            "false_color": str(false_color_path)
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    EVENT_ID = 84118

    LATITUDE = 19.6374
    LONGITUDE = 74.9979

    result = generate_sentinel_visualizations(
        event_id=EVENT_ID,
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    print("\n" + "=" * 60)
    print("SENTINEL VISUALIZATION TEST")
    print("=" * 60)

    print(f"Event: {EVENT_ID}")

    for name, path in result["visualizations"].items():
        print(f"{name}: {path}")

    print("=" * 60)