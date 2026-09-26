from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform


# ============================================================
# FIND SENTINEL BAND
# ============================================================

def find_sentinel_band(event_id, band_name):
    """
    Find a Sentinel-2 band from the local event cache.
    """

    project_root = Path(__file__).resolve().parents[2]

    sentinel_dir = (
        project_root
        / "data"
        / "sentinel"
        / f"event_{event_id}"
    )

    matches = list(
        sentinel_dir.glob(
            f"*_{band_name}.jp2"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"{band_name} not found "
            f"for Event {event_id}"
        )

    return matches[0]


# ============================================================
# CALCULATE SENTINEL INDICES
# ============================================================

def calculate_sentinel_indices(
    latitude,
    longitude,
    event_id,
    buffer_m=500
):
    """
    Calculate NDVI, NBR and SCL information
    around a FIRMS hotspot.

    B04/B08 = 10 m
    B12/SCL  = 20 m

    B12 and SCL are resampled to the
    B04 10 m grid.
    """

    # --------------------------------------------------------
    # FIND FILES
    # --------------------------------------------------------

    b04_file = find_sentinel_band(
        event_id,
        "B04_10m"
    )

    b08_file = find_sentinel_band(
        event_id,
        "B08_10m"
    )

    b12_file = find_sentinel_band(
        event_id,
        "B12_20m"
    )

    scl_file = find_sentinel_band(
        event_id,
        "SCL_20m"
    )

    # --------------------------------------------------------
    # OPEN B04
    # --------------------------------------------------------

    with rasterio.open(b04_file) as b04_src:

        # Convert hotspot coordinates
        # EPSG:4326 -> Sentinel CRS

        x, y = transform(
            "EPSG:4326",
            b04_src.crs,
            [longitude],
            [latitude]
        )

        x = x[0]
        y = y[0]

        # Find hotspot pixel

        hotspot_row, hotspot_col = (
            b04_src.index(x, y)
        )

        resolution = abs(
            b04_src.transform.a
        )

        # Buffer in pixels

        buffer_pixels = int(
            buffer_m / resolution
        )

        row_start = max(
            0,
            hotspot_row - buffer_pixels
        )

        row_stop = min(
            b04_src.height,
            hotspot_row + buffer_pixels
        )

        col_start = max(
            0,
            hotspot_col - buffer_pixels
        )

        col_stop = min(
            b04_src.width,
            hotspot_col + buffer_pixels
        )

        window = rasterio.windows.Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start
        )

        # Read B04

        b04 = b04_src.read(
            1,
            window=window
        ).astype(
            np.float32
        )

        window_transform = (
            b04_src.window_transform(window)
        )

        target_crs = b04_src.crs

        target_height = b04.shape[0]
        target_width = b04.shape[1]

    # --------------------------------------------------------
    # READ B08
    # --------------------------------------------------------

    with rasterio.open(b08_file) as b08_src:

        b08 = b08_src.read(
            1,
            window=window
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # READ B12
    # --------------------------------------------------------
    #
    # B12 = 20 m
    #
    # Reproject directly to the
    # B04 10 m grid.
    # --------------------------------------------------------

    with rasterio.open(b12_file) as b12_src:

        b12 = np.empty(
            (
                target_height,
                target_width
            ),
            dtype=np.float32
        )

        rasterio.warp.reproject(
            source=rasterio.band(
                b12_src,
                1
            ),
            destination=b12,
            src_transform=b12_src.transform,
            src_crs=b12_src.crs,
            dst_transform=window_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

    # --------------------------------------------------------
    # READ SCL
    # --------------------------------------------------------
    #
    # SCL is categorical.
    #
    # Therefore use nearest-neighbour
    # instead of bilinear.
    # --------------------------------------------------------

    with rasterio.open(scl_file) as scl_src:

        scl = np.empty(
            (
                target_height,
                target_width
            ),
            dtype=np.uint8
        )

        rasterio.warp.reproject(
            source=rasterio.band(
                scl_src,
                1
            ),
            destination=scl,
            src_transform=scl_src.transform,
            src_crs=scl_src.crs,
            dst_transform=window_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    # --------------------------------------------------------
    # CONVERT TO REFLECTANCE
    # --------------------------------------------------------

    b04 = b04 / 10000.0
    b08 = b08 / 10000.0
    b12 = b12 / 10000.0

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    ndvi_denominator = (
        b08 + b04
    )

    ndvi = np.divide(
        b08 - b04,
        ndvi_denominator,
        out=np.zeros_like(b08),
        where=ndvi_denominator != 0
    )

    # --------------------------------------------------------
    # NBR
    # --------------------------------------------------------

    nbr_denominator = (
        b08 + b12
    )

    nbr = np.divide(
        b08 - b12,
        nbr_denominator,
        out=np.zeros_like(b08),
        where=nbr_denominator != 0
    )

    # --------------------------------------------------------
    # HOTSPOT PIXEL INSIDE WINDOW
    # --------------------------------------------------------

    hotspot_row_local = (
        hotspot_row - row_start
    )

    hotspot_col_local = (
        hotspot_col - col_start
    )

    # Make sure indices are inside array

    hotspot_row_local = max(
        0,
        min(
            hotspot_row_local,
            ndvi.shape[0] - 1
        )
    )

    hotspot_col_local = max(
        0,
        min(
            hotspot_col_local,
            ndvi.shape[1] - 1
        )
    )

    center_ndvi = float(
        ndvi[
            hotspot_row_local,
            hotspot_col_local
        ]
    )

    center_nbr = float(
        nbr[
            hotspot_row_local,
            hotspot_col_local
        ]
    )

    center_scl = int(
        scl[
            hotspot_row_local,
            hotspot_col_local
        ]
    )

    # --------------------------------------------------------
    # SCL CLASS COUNTS
    # --------------------------------------------------------

    unique_classes, counts = np.unique(
        scl,
        return_counts=True
    )

    scl_classes = {
        str(int(cls)): int(count)
        for cls, count in zip(
            unique_classes,
            counts
        )
    }

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "event_id": int(event_id),

        "location": {
            "latitude": float(latitude),
            "longitude": float(longitude)
        },

        "buffer_m": buffer_m,

        "array_shape": {
            "height": int(ndvi.shape[0]),
            "width": int(ndvi.shape[1])
        },

        "ndvi": {
            "min": float(
                np.nanmin(ndvi)
            ),
            "mean": float(
                np.nanmean(ndvi)
            ),
            "max": float(
                np.nanmax(ndvi)
            ),
            "center": center_ndvi
        },

        "nbr": {
            "min": float(
                np.nanmin(nbr)
            ),
            "mean": float(
                np.nanmean(nbr)
            ),
            "max": float(
                np.nanmax(nbr)
            ),
            "center": center_nbr
        },

        "scl": {
            "center_class": center_scl,
            "classes": scl_classes
        },

        "files": {
            "B04": str(b04_file),
            "B08": str(b08_file),
            "B12": str(b12_file),
            "SCL": str(scl_file)
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = calculate_sentinel_indices(
        latitude=19.6374,
        longitude=74.9979,
        event_id=84118,
        buffer_m=500
    )

    print()
    print("=" * 70)
    print("SENTINEL EVIDENCE TEST")
    print("=" * 70)

    print(
        "Event:",
        result["event_id"]
    )

    print(
        "NDVI:",
        result["ndvi"]
    )

    print(
        "NBR:",
        result["nbr"]
    )

    print(
        "SCL:",
        result["scl"]
    )

    print(
        "Array:",
        result["array_shape"]
    )

    print("=" * 70)