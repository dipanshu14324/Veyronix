r"""
Sentinel-2 Corpus Builder
=========================

Purpose
-------
Build an Event -> Sentinel-2 L2A product index for FIRMS events.

This stage:
1. Reads FIRMS event-level data.
2. Searches Copernicus Data Space STAC.
3. Finds Sentinel-2 L2A scenes around each event.
4. Selects the best scene.
5. Creates event -> Sentinel product mapping.
6. Deduplicates Sentinel products.

IMPORTANT
---------
This script ONLY searches Sentinel metadata.

It does NOT download:
    B04
    B08
    B12
    SCL

Those downloads will be handled in the next stage.

Input
-----
data/ml_ready/phase1_event_level_ml_dataset.csv

Output
------
data/sentinel/
    sentinel_event_index.csv
    sentinel_products.csv
    corpus_progress.json

Run
---
Process events 0 to 10:

    python geo\sentinel_corpus.py 0 10

Process events 400 to 800:

    python geo\sentinel_corpus.py 400 800

Process events 800 to 1200:

    python geo\sentinel_corpus.py 800 1200

Process complete corpus:

    python geo\sentinel_corpus.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVENT_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "sentinel"
)

EVENT_INDEX_FILE = (
    OUTPUT_DIR
    / "sentinel_event_index.csv"
)

PRODUCT_FILE = (
    OUTPUT_DIR
    / "sentinel_products.csv"
)

PROGRESS_FILE = (
    OUTPUT_DIR
    / "corpus_progress.json"
)


# ============================================================
# COPERNICUS DATA SPACE STAC
# ============================================================

STAC_SEARCH_URL = (
    "https://stac.dataspace.copernicus.eu/v1/search"
)

STAC_COLLECTION = (
    "sentinel-2-l2a"
)


# ============================================================
# SEARCH SETTINGS
# ============================================================

TIME_WINDOW_DAYS = 5

MAX_CLOUD_COVER = 60.0

MAX_RESULTS = 20

REQUEST_TIMEOUT = 60

MAX_RETRIES = 5

RETRY_SLEEP_SECONDS = 3

CHECKPOINT_EVERY = 100


# ============================================================
# REQUIRED EVENT COLUMNS
# ============================================================

REQUIRED_EVENT_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(
    "sentinel_corpus"
)


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value: Any,
) -> Optional[float]:

    try:

        if pd.isna(value):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# ACQ TIME
# ============================================================

def parse_acq_time(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    value = str(value).strip()

    if not value:
        return None

    if value.endswith(".0"):
        value = value[:-2]

    # --------------------------------------------------------
    # Colon format
    # --------------------------------------------------------

    if ":" in value:

        parts = value.split(":")

        try:

            if len(parts) == 2:

                hour = int(parts[0])
                minute = int(parts[1])
                second = 0

            elif len(parts) >= 3:

                hour = int(parts[0])
                minute = int(parts[1])
                second = int(parts[2])

            else:

                return None

            if (
                hour > 23
                or minute > 59
                or second > 59
            ):

                return None

            return (
                f"{hour:02d}:"
                f"{minute:02d}:"
                f"{second:02d}"
            )

        except ValueError:

            return None

    # --------------------------------------------------------
    # Numeric format
    # --------------------------------------------------------

    value = "".join(
        ch
        for ch in value
        if ch.isdigit()
    )

    if not value:
        return None

    try:

        if len(value) <= 4:

            value = value.zfill(4)

            hour = int(
                value[:2]
            )

            minute = int(
                value[2:4]
            )

            second = 0

        elif len(value) == 6:

            hour = int(
                value[:2]
            )

            minute = int(
                value[2:4]
            )

            second = int(
                value[4:6]
            )

        else:

            value = value[:6]

            hour = int(
                value[:2]
            )

            minute = int(
                value[2:4]
            )

            second = int(
                value[4:6]
            )

        if (
            hour > 23
            or minute > 59
            or second > 59
        ):

            return None

        return (
            f"{hour:02d}:"
            f"{minute:02d}:"
            f"{second:02d}"
        )

    except ValueError:

        return None


# ============================================================
# EVENT DATETIME
# ============================================================

def build_event_datetime(
    row: pd.Series,
) -> Optional[datetime]:

    try:

        date_value = pd.to_datetime(
            row["acq_date"],
            errors="coerce",
        )

        if pd.isna(date_value):
            return None

        time_value = None

        if "acq_time" in row.index:

            time_value = parse_acq_time(
                row.get("acq_time")
            )

        if time_value is not None:

            hour, minute, second = map(
                int,
                time_value.split(":"),
            )

        else:

            # Current dataset does not have acq_time.
            hour = 12
            minute = 0
            second = 0

        return datetime(
            year=date_value.year,
            month=date_value.month,
            day=date_value.day,
            hour=hour,
            minute=minute,
            second=second,
            tzinfo=timezone.utc,
        )

    except Exception:

        return None


# ============================================================
# ISO FORMAT
# ============================================================

def iso_z(
    dt: datetime,
) -> str:

    return (
        dt.astimezone(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )


# ============================================================
# GEOJSON POINT
# ============================================================

def make_point(
    latitude: float,
    longitude: float,
) -> Dict[str, Any]:

    return {
        "type": "Point",
        "coordinates": [
            longitude,
            latitude,
        ],
    }


# ============================================================
# STAC SEARCH
# ============================================================

def stac_search(
    latitude: float,
    longitude: float,
    event_datetime: datetime,
) -> List[Dict[str, Any]]:

    start_dt = (
        event_datetime
        - timedelta(
            days=TIME_WINDOW_DAYS
        )
    )

    end_dt = (
        event_datetime
        + timedelta(
            days=TIME_WINDOW_DAYS
        )
    )

    payload = {

        "collections": [
            STAC_COLLECTION
        ],

        "datetime": (
            f"{iso_z(start_dt)}/"
            f"{iso_z(end_dt)}"
        ),

        "intersects": make_point(
            latitude,
            longitude,
        ),

        "query": {

            "eo:cloud_cover": {

                "lte":
                    MAX_CLOUD_COVER

            }

        },

        "limit": MAX_RESULTS,

        "sortby": [

            {
                "field": "datetime",
                "direction": "asc",
            }

        ],

        "fields": {

            "exclude": [
                "geometry"
            ]

        },
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.post(
                STAC_SEARCH_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Content-Type":
                        "application/json",

                    "Accept":
                        "application/geo+json",
                },
            )

            response.raise_for_status()

            data = response.json()

            return data.get(
                "features",
                [],
            )

        except requests.RequestException as exc:

            logger.warning(
                "STAC request failed "
                "(attempt %s/%s): %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_SLEEP_SECONDS
                )

            else:

                raise

    return []


# ============================================================
# EXTRACT STAC ITEM
# ============================================================

def extract_item_metadata(
    item: Dict[str, Any],
    event_datetime: datetime,
) -> Dict[str, Any]:

    properties = item.get(
        "properties",
        {},
    )

    product_id = item.get(
        "id"
    )

    product_datetime = (
        properties.get(
            "datetime"
        )
        or properties.get(
            "start_datetime"
        )
    )

    cloud_cover = properties.get(
        "eo:cloud_cover"
    )

    platform = properties.get(
        "platform"
    )

    constellation = properties.get(
        "constellation"
    )

    processing_level = properties.get(
        "processing:level"
    )

    # --------------------------------------------------------
    # MGRS tile
    # --------------------------------------------------------

    mgrs_tile = (
        properties.get(
            "s2:mgrs_tile"
        )
        or properties.get(
            "mgrs:utm_zone"
        )
    )

    # Fallback: extract from product name.
    if not mgrs_tile and product_id:

        parts = str(
            product_id
        ).split("_")

        for part in parts:

            if (
                part.startswith("T")
                and len(part) >= 6
            ):

                mgrs_tile = part[1:]
                break

    # --------------------------------------------------------
    # Product datetime
    # --------------------------------------------------------

    parsed_product_datetime = None

    if product_datetime:

        try:

            parsed_product_datetime = pd.to_datetime(
                product_datetime,
                utc=True,
                errors="coerce",
            )

        except Exception:

            parsed_product_datetime = None

    # --------------------------------------------------------
    # Time difference
    # --------------------------------------------------------

    time_difference_hours = None

    if (
        parsed_product_datetime is not None
        and not pd.isna(
            parsed_product_datetime
        )
    ):

        event_timestamp = pd.Timestamp(
            event_datetime
        )

        difference = (
            parsed_product_datetime
            - event_timestamp
        )

        time_difference_hours = (
            abs(
                difference.total_seconds()
            )
            / 3600.0
        )

    # --------------------------------------------------------
    # Product name
    # --------------------------------------------------------

    product_name = properties.get(
        "title"
    )

    if product_name is None:

        product_name = product_id

    # --------------------------------------------------------
    # Assets
    # --------------------------------------------------------

    assets = item.get(
        "assets",
        {}
    )

    asset_names = list(
        assets.keys()
    )

    required_assets = [
        "B04_10m",
        "B08_10m",
        "B12_20m",
        "SCL_20m",
    ]

    available_required_assets = [
        asset
        for asset in required_assets
        if asset in assets
    ]

    # --------------------------------------------------------
    # Self link
    # --------------------------------------------------------

    self_link = None

    for link in item.get(
        "links",
        [],
    ):

        if link.get(
            "rel"
        ) == "self":

            self_link = link.get(
                "href"
            )

            break

    return {

        "product_id":
            product_id,

        "product_name":
            product_name,

        "product_datetime":
            product_datetime,

        "cloud_cover":
            safe_float(
                cloud_cover
            ),

        "platform":
            platform,

        "constellation":
            constellation,

        "processing_level":
            processing_level,

        "mgrs_tile":
            mgrs_tile,

        "time_difference_hours":
            time_difference_hours,

        "asset_count":
            len(
                asset_names
            ),

        "required_assets_available":
            ",".join(
                available_required_assets
            ),

        "asset_names":
            ",".join(
                asset_names
            ),

        "self_link":
            self_link,
    }


# ============================================================
# SELECT BEST PRODUCT
# ============================================================

def select_best_product(
    items: List[Dict[str, Any]],
    event_datetime: datetime,
) -> Optional[Dict[str, Any]]:

    if not items:
        return None

    candidates = []

    for item in items:

        metadata = extract_item_metadata(
            item,
            event_datetime,
        )

        time_difference = metadata.get(
            "time_difference_hours"
        )

        cloud_cover = metadata.get(
            "cloud_cover"
        )

        if time_difference is None:
            time_difference = float("inf")

        if cloud_cover is None:
            cloud_cover = float("inf")

        candidates.append(
            (
                time_difference,
                cloud_cover,
                metadata,
            )
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return candidates[0][2]


# ============================================================
# PROGRESS
# ============================================================

def load_progress() -> Dict[str, Any]:

    if not PROGRESS_FILE.exists():

        return {
            "last_index": 0,
            "processed": 0,
            "success": 0,
            "no_scene": 0,
            "errors": 0,
        }

    try:

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {
            "last_index": 0,
            "processed": 0,
            "success": 0,
            "no_scene": 0,
            "errors": 0,
        }


def save_progress(
    progress: Dict[str, Any],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = (
        PROGRESS_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            progress,
            file,
            indent=2,
        )

    os.replace(
        temp_file,
        PROGRESS_FILE,
    )


# ============================================================
# LOAD EXISTING EVENT INDEX
# ============================================================

def load_existing_event_index() -> pd.DataFrame:

    if not EVENT_INDEX_FILE.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            EVENT_INDEX_FILE
        )

    except Exception as exc:

        logger.warning(
            "Could not read event index: %s",
            exc,
        )

        return pd.DataFrame()


# ============================================================
# LOAD EXISTING PRODUCT INDEX
# ============================================================

def load_existing_products() -> pd.DataFrame:

    if not PRODUCT_FILE.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            PRODUCT_FILE
        )

    except Exception as exc:

        logger.warning(
            "Could not read product index: %s",
            exc,
        )

        return pd.DataFrame()


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    event_records: List[Dict[str, Any]],
    product_records: List[Dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if event_records:

        event_df = pd.DataFrame(
            event_records
        )

        if "event_id" in event_df.columns:

            event_df = (
                event_df
                .drop_duplicates(
                    subset=[
                        "event_id"
                    ],
                    keep="last",
                )
            )

        event_df.to_csv(
            EVENT_INDEX_FILE,
            index=False,
        )

    if product_records:

        product_df = pd.DataFrame(
            product_records
        )

        if "product_id" in product_df.columns:

            product_df = (
                product_df
                .drop_duplicates(
                    subset=[
                        "product_id"
                    ],
                    keep="last",
                )
            )

        product_df.to_csv(
            PRODUCT_FILE,
            index=False,
        )


# ============================================================
# BUILD CORPUS
# ============================================================

def build_corpus(
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:

    # --------------------------------------------------------
    # Load events
    # --------------------------------------------------------

    logger.info(
        "Loading FIRMS event dataset..."
    )

    if not EVENT_FILE.exists():

        raise FileNotFoundError(
            f"Event file not found:\n"
            f"{EVENT_FILE}"
        )

    events = pd.read_csv(
        EVENT_FILE
    )

    logger.info(
        "Loaded %s FIRMS events.",
        len(events),
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_EVENT_COLUMNS
        if column not in events.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    total_events = len(events)

    # --------------------------------------------------------
    # Correct range handling
    #
    # 0 10   = rows 0..9
    # 400 800 = rows 400..799
    # --------------------------------------------------------

    start_index = max(
        0,
        start_index
    )

    if end_index is None:

        end_index = total_events

    end_index = min(
        end_index,
        total_events
    )

    if start_index >= end_index:

        raise ValueError(
            f"Invalid range: "
            f"{start_index} -> "
            f"{end_index}"
        )

    # --------------------------------------------------------
    # Existing data
    # --------------------------------------------------------

    existing_events = (
        load_existing_event_index()
    )

    existing_products = (
        load_existing_products()
    )

    event_records = []

    product_records = []

    processed_event_ids = set()

    # --------------------------------------------------------
    # Preserve existing events
    # --------------------------------------------------------

    if not existing_events.empty:

        event_records.extend(
            existing_events.to_dict(
                orient="records"
            )
        )

        if "event_id" in existing_events.columns:

            processed_event_ids = set(
                existing_events[
                    "event_id"
                ]
                .astype(str)
            )

    # --------------------------------------------------------
    # Preserve existing products
    # --------------------------------------------------------

    if not existing_products.empty:

        product_records.extend(
            existing_products.to_dict(
                orient="records"
            )
        )

    # --------------------------------------------------------
    # Existing product IDs
    # --------------------------------------------------------

    product_ids = set()

    for record in product_records:

        product_id = record.get(
            "product_id"
        )

        if product_id:

            product_ids.add(
                str(product_id)
            )

    logger.info(
        "Processing events %s -> %s",
        start_index,
        end_index,
    )

    # --------------------------------------------------------
    # Progress counters for THIS run
    # --------------------------------------------------------

    run_processed = 0
    run_success = 0
    run_no_scene = 0
    run_errors = 0

    # --------------------------------------------------------
    # Process events
    # --------------------------------------------------------

    for index in range(
        start_index,
        end_index,
    ):

        row = events.iloc[
            index
        ]

        event_id = row.get(
            "event_id",
            index,
        )

        event_id_string = str(
            event_id
        )

        # ----------------------------------------------------
        # Resume / skip
        # ----------------------------------------------------

        if (
            event_id_string
            in processed_event_ids
        ):

            continue

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        latitude = safe_float(
            row["latitude"]
        )

        longitude = safe_float(
            row["longitude"]
        )

        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        event_datetime = (
            build_event_datetime(
                row
            )
        )

        # ----------------------------------------------------
        # Invalid event
        # ----------------------------------------------------

        if (
            latitude is None
            or longitude is None
            or event_datetime is None
        ):

            event_records.append(
                {
                    "event_id":
                        event_id,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "acq_date":
                        row.get(
                            "acq_date"
                        ),

                    "acq_time":
                        row.get(
                            "acq_time",
                            None,
                        ),

                    "sentinel_available":
                        False,

                    "status":
                        "invalid_event",
                }
            )

            run_processed += 1
            run_errors += 1

            processed_event_ids.add(
                event_id_string
            )

            continue

        # ----------------------------------------------------
        # STAC SEARCH
        # ----------------------------------------------------

        try:

            items = stac_search(
                latitude=latitude,
                longitude=longitude,
                event_datetime=event_datetime,
            )

            best = select_best_product(
                items,
                event_datetime,
            )

            # ------------------------------------------------
            # No scene
            # ------------------------------------------------

            if best is None:

                event_records.append(
                    {
                        "event_id":
                            event_id,

                        "latitude":
                            latitude,

                        "longitude":
                            longitude,

                        "acq_date":
                            row.get(
                                "acq_date"
                            ),

                        "acq_time":
                            row.get(
                                "acq_time",
                                None,
                            ),

                        "event_datetime_utc":
                            event_datetime.isoformat(),

                        "sentinel_available":
                            False,

                        "status":
                            "no_scene",

                        "search_window_days":
                            TIME_WINDOW_DAYS,

                        "max_cloud_cover":
                            MAX_CLOUD_COVER,
                    }
                )

                run_processed += 1
                run_no_scene += 1

                processed_event_ids.add(
                    event_id_string
                )

                continue

            # ------------------------------------------------
            # Matched
            # ------------------------------------------------

            event_record = {

                "event_id":
                    event_id,

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "acq_date":
                    row.get(
                        "acq_date"
                    ),

                "acq_time":
                    row.get(
                        "acq_time",
                        None,
                    ),

                "event_datetime_utc":
                    event_datetime.isoformat(),

                "sentinel_available":
                    True,

                "status":
                    "matched",

                "sentinel_product_id":
                    best.get(
                        "product_id"
                    ),

                "sentinel_product_name":
                    best.get(
                        "product_name"
                    ),

                "sentinel_datetime":
                    best.get(
                        "product_datetime"
                    ),

                "sentinel_cloud_cover":
                    best.get(
                        "cloud_cover"
                    ),

                "sentinel_platform":
                    best.get(
                        "platform"
                    ),

                "sentinel_tile":
                    best.get(
                        "mgrs_tile"
                    ),

                "sentinel_time_difference_hours":
                    best.get(
                        "time_difference_hours"
                    ),

                "sentinel_required_assets":
                    best.get(
                        "required_assets_available"
                    ),
            }

            event_records.append(
                event_record
            )

            # ------------------------------------------------
            # Add unique product
            # ------------------------------------------------

            product_id = best.get(
                "product_id"
            )

            if (
                product_id
                and str(product_id)
                not in product_ids
            ):

                product_record = {

                    "product_id":
                        product_id,

                    "product_name":
                        best.get(
                            "product_name"
                        ),

                    "product_datetime":
                        best.get(
                            "product_datetime"
                        ),

                    "cloud_cover":
                        best.get(
                            "cloud_cover"
                        ),

                    "platform":
                        best.get(
                            "platform"
                        ),

                    "constellation":
                        best.get(
                            "constellation"
                        ),

                    "processing_level":
                        best.get(
                            "processing_level"
                        ),

                    "mgrs_tile":
                        best.get(
                            "mgrs_tile"
                        ),

                    "required_assets":
                        best.get(
                            "required_assets_available"
                        ),

                    "asset_count":
                        best.get(
                            "asset_count"
                        ),

                    "self_link":
                        best.get(
                            "self_link"
                        ),

                    "downloaded":
                        False,
                }

                product_records.append(
                    product_record
                )

                product_ids.add(
                    str(product_id)
                )

            run_processed += 1
            run_success += 1

            processed_event_ids.add(
                event_id_string
            )

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        except Exception as exc:

            logger.exception(
                "Event %s failed.",
                event_id,
            )

            event_records.append(
                {
                    "event_id":
                        event_id,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "acq_date":
                        row.get(
                            "acq_date"
                        ),

                    "acq_time":
                        row.get(
                            "acq_time",
                            None,
                        ),

                    "sentinel_available":
                        False,

                    "status":
                        "error",

                    "error":
                        str(exc),
                }
            )

            run_processed += 1
            run_errors += 1

            processed_event_ids.add(
                event_id_string
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        current_position = index + 1

        if (
            run_processed
            % CHECKPOINT_EVERY
            == 0
        ):

            save_outputs(
                event_records,
                product_records,
            )

            logger.info(
                "Progress: %s/%s | "
                "run_processed=%s | "
                "matched=%s | "
                "no_scene=%s | "
                "errors=%s | "
                "unique_products=%s",

                current_position,

                end_index,

                run_processed,

                run_success,

                run_no_scene,

                run_errors,

                len(product_ids),
            )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    save_outputs(
        event_records,
        product_records,
    )

    # --------------------------------------------------------
    # Progress JSON
    # --------------------------------------------------------

    progress = {
        "last_index":
            end_index,

        "processed_this_run":
            run_processed,

        "matched_this_run":
            run_success,

        "no_scene_this_run":
            run_no_scene,

        "errors_this_run":
            run_errors,

        "total_events":
            total_events,

        "unique_products":
            len(product_ids),

        "updated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    save_progress(
        progress
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("")

    logger.info(
        "=" * 70
    )

    logger.info(
        "SENTINEL CORPUS BUILD COMPLETE"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Total FIRMS events: %s",
        total_events,
    )

    logger.info(
        "Requested range: %s -> %s",
        start_index,
        end_index,
    )

    logger.info(
        "Processed this run: %s",
        run_processed,
    )

    logger.info(
        "Matched this run: %s",
        run_success,
    )

    logger.info(
        "No Sentinel scene this run: %s",
        run_no_scene,
    )

    logger.info(
        "Errors this run: %s",
        run_errors,
    )

    logger.info(
        "Total unique Sentinel products: %s",
        len(product_ids),
    )

    logger.info(
        "Event index: %s",
        EVENT_INDEX_FILE,
    )

    logger.info(
        "Product index: %s",
        PRODUCT_FILE,
    )

    logger.info(
        "Progress: %s",
        PROGRESS_FILE,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SENTINEL-2 CORPUS BUILDER")
    print("=" * 70)
    print()

    print("Input:")
    print(EVENT_FILE)
    print()

    print("STAC:")
    print(STAC_SEARCH_URL)
    print()

    print(
        f"Time window: +/- {TIME_WINDOW_DAYS} days"
    )

    print(
        f"Maximum cloud cover: "
        f"{MAX_CLOUD_COVER}%"
    )

    print()

    print("IMPORTANT:")
    print(
        "This stage searches metadata only."
    )

    print(
        "No Sentinel JP2 files will be downloaded."
    )

    print()

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not EVENT_FILE.exists():

        print(
            "ERROR: FIRMS event dataset not found."
        )

        print(
            EVENT_FILE
        )

        sys.exit(1)

    # --------------------------------------------------------
    # COMMAND FORMAT
    #
    # python geo\sentinel_corpus.py START END
    #
    # Examples:
    #
    # 0 10
    # 400 800
    # 800 1200
    #
    # The second argument is the END INDEX,
    # NOT the number of events.
    # --------------------------------------------------------

    start_index = 0
    end_index = None

    if len(sys.argv) >= 2:

        try:

            start_index = int(
                sys.argv[1]
            )

        except ValueError:

            print(
                "ERROR: Invalid start index."
            )

            sys.exit(1)

    if len(sys.argv) >= 3:

        try:

            end_index = int(
                sys.argv[2]
            )

        except ValueError:

            print(
                "ERROR: Invalid end index."
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Validate range before running
    # --------------------------------------------------------

    if start_index < 0:

        print(
            "ERROR: Start index cannot be negative."
        )

        sys.exit(1)

    if end_index is not None:

        if end_index <= start_index:

            print(
                "ERROR: End index must be "
                "greater than start index."
            )

            print(
                f"Received: "
                f"{start_index} -> {end_index}"
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    build_corpus(
        start_index=start_index,
        end_index=end_index,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()