
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from functools import lru_cache
import pandas as pd
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.append(str(BASE_DIR))

from ml.predict_event import predict_event


try:
    from geo.sentinel_manager import (
        get_sentinel_status,
        get_sentinel_files,
    )
except Exception:
    get_sentinel_status = None
    get_sentinel_files = None


try:
    from geo.geo.sentinel_map import (
        calculate_sentinel_indices,
    )
except Exception:
    calculate_sentinel_indices = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="VEYRONIX AI",
    description=(
        "VEYRONIX AI - Industrial Fire Detection and "
        "Thermal Source Attribution using NASA FIRMS, "
        "LightGBM and Sentinel-2 evidence."
    ),
    version="2.0.0",
)


# ============================================================
# SCHEMAS
# ============================================================

class ThermalEvent(BaseModel):
    observation_count: float = 0
    mean_frp: float = 0
    peak_frp: float = 0
    total_frp: float = 0
    mean_brightness: float = 0
    persistent: float = 0
    historical_detection_count: float = 0
    historical_mean_frp: float = 0
    local_frp_deviation: float = 0
    historical_daily_activity: float = 0
    previously_detected: float = 0
    anomaly_score: float = 0
    is_anomaly: float = 0


class EventRequest(BaseModel):
    event_id: int


class BatchEventRequest(BaseModel):
    event_ids: list[int]


# ============================================================
# DATA LOADER
# ============================================================

@lru_cache(maxsize=1)
def load_event_data():
    """
    Loads the ML-ready event dataset once per process.

    Used by individual-event endpoints.

    The /events endpoint intentionally does NOT use this
    function because it uses chunked CSV processing.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if "event_id" in df.columns:
        df["event_id"] = pd.to_numeric(
            df["event_id"],
            errors="coerce"
        )

    return df


# ============================================================
# HELPERS
# ============================================================

def create_map_links(latitude, longitude):
    return {
        "google_maps": (
            "https://www.google.com/maps/search/?api=1"
            f"&query={latitude},{longitude}"
        ),
        "openstreetmap": (
            "https://www.openstreetmap.org/"
            f"?mlat={latitude}"
            f"&mlon={longitude}"
            f"&zoom=15"
        ),
    }


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default

        return int(float(value))

    except Exception:
        return default


def get_feature_dict(row):
    feature_names = [
        "observation_count",
        "mean_frp",
        "peak_frp",
        "total_frp",
        "mean_brightness",
        "persistent",
        "historical_detection_count",
        "historical_mean_frp",
        "local_frp_deviation",
        "historical_daily_activity",
        "previously_detected",
        "anomaly_score",
        "is_anomaly",
    ]

    return {
        name: safe_float(
            row.get(name, 0)
        )
        for name in feature_names
    }


def serialize_event(row):
    latitude = safe_float(
        row.get("latitude")
    )

    longitude = safe_float(
        row.get("longitude")
    )

    return {
        "event_id": safe_int(
            row.get("event_id")
        ),

        "latitude": latitude,

        "longitude": longitude,

        "event_date": str(
            row.get("event_date", "")
        ),

        "start_time": str(
            row.get("start_time", "")
        ),

        "end_time": str(
            row.get("end_time", "")
        ),

        "observation_count": safe_int(
            row.get("observation_count")
        ),

        "mean_frp": safe_float(
            row.get("mean_frp")
        ),

        "peak_frp": safe_float(
            row.get("peak_frp")
        ),

        "total_frp": safe_float(
            row.get("total_frp")
        ),

        "mean_brightness": safe_float(
            row.get("mean_brightness")
        ),

        "persistent": safe_float(
            row.get("persistent")
        ),

        "historical_detection_count": safe_float(
            row.get("historical_detection_count")
        ),

        "historical_mean_frp": safe_float(
            row.get("historical_mean_frp")
        ),

        "local_frp_deviation": safe_float(
            row.get("local_frp_deviation")
        ),

        "historical_daily_activity": safe_float(
            row.get("historical_daily_activity")
        ),

        "previously_detected": safe_float(
            row.get("previously_detected")
        ),

        "anomaly_score": safe_float(
            row.get("anomaly_score")
        ),

        "is_anomaly": bool(
            safe_int(
                row.get("is_anomaly")
            )
        ),

        "maps": create_map_links(
            latitude,
            longitude
        ),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():
    return {
        "status": "success",
        "service": "VEYRONIX AI",
        "problem_statement": "SIH26162",
        "message": (
            "VEYRONIX AI backend is running."
        ),
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    dataset_available = DATA_FILE.exists()

    model_available = False

    try:
        model_available = callable(
            predict_event
        )
    except Exception:
        model_available = False

    return {
        "status": "healthy",
        "service": "VEYRONIX AI",
        "model_available": model_available,
        "dataset_available": dataset_available,
        "api_key_configured": False,
        "authentication": (
            "disabled_for_local_SIH_demo"
        ),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "LightGBM",

        "task": (
            "Thermal Source Classification"
        ),

        "classes": [
            "Agriculture_Biomass",
            "Forest_Natural",
            "Industrial",
            "Waste_Other",
        ],

        "anomaly_detector": (
            "Isolation Forest"
        ),

        "sentinel": {
            "enabled": True,
            "cache": True,
            "bands": [
                "B04",
                "B08",
                "B12",
                "SCL",
            ],
        },
    }


# ============================================================
# MANUAL PREDICTION
# ============================================================

@app.post("/predict")
def predict(event: ThermalEvent):

    try:

        result = predict_event(
            event.model_dump()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"ML prediction failed: {exc}"
            ),
        )

    return {
        "status": "success",

        "prediction": {
            "source": result[
                "predicted_source"
            ],

            "confidence": result[
                "confidence"
            ],
        },

        "probabilities": result[
            "probabilities"
        ],

        "anomaly": {
            "is_anomaly": bool(
                event.is_anomaly
            ),

            "anomaly_score": (
                event.anomaly_score
            ),
        },
    }


# ============================================================
# EVENTS
# RENDER-SAFE CHUNKED IMPLEMENTATION
# ============================================================

@app.get("/events")
def events(
    date: str | None = None,
    min_frp: float | None = None,
    max_frp: float | None = None,
    anomaly: bool | None = None,
    persistent: bool | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    limit: int = 100,
    sort_by: str = "latest",
):
    """
    Render-safe /events endpoint.

    The 91 MB CSV is NOT loaded into one giant pandas
    DataFrame.

    Instead the file is processed in 50,000-row chunks.
    Only the rows needed for the final response are retained.
    """

    try:

        # ----------------------------------------------------
        # LIMIT
        # ----------------------------------------------------

        limit = max(
            1,
            min(int(limit), 300)
        )

        # ----------------------------------------------------
        # SORT MODE
        # ----------------------------------------------------

        if sort_by not in {
            "latest",
            "frp",
        }:
            sort_by = "latest"

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "event_id",
            "latitude",
            "longitude",
            "event_date",
            "start_time",
            "end_time",
            "observation_count",
            "mean_frp",
            "peak_frp",
            "total_frp",
            "mean_brightness",
            "persistent",
            "historical_detection_count",
            "historical_mean_frp",
            "local_frp_deviation",
            "historical_daily_activity",
            "previously_detected",
            "anomaly_score",
            "is_anomaly",
        ]

        # ----------------------------------------------------
        # SMALL RESULT COLLECTION
        # ----------------------------------------------------

        collected = []

        # ----------------------------------------------------
        # CHUNKED CSV READING
        # ----------------------------------------------------

        for chunk in pd.read_csv(
            DATA_FILE,
            usecols=required_columns,
            chunksize=50000,
        ):

            # ------------------------------------------------
            # EVENT ID
            # ------------------------------------------------

            chunk["event_id"] = pd.to_numeric(
                chunk["event_id"],
                errors="coerce"
            )

            # ------------------------------------------------
            # DATE
            # ------------------------------------------------

            if date:

                chunk = chunk[
                    chunk["event_date"]
                    .astype(str)
                    .str.startswith(date)
                ]

            # ------------------------------------------------
            # FRP
            # ------------------------------------------------

            if (
                min_frp is not None
                or max_frp is not None
                or sort_by == "frp"
            ):

                chunk["peak_frp"] = pd.to_numeric(
                    chunk["peak_frp"],
                    errors="coerce"
                ).fillna(0)

            if min_frp is not None:

                chunk = chunk[
                    chunk["peak_frp"]
                    >= min_frp
                ]

            if max_frp is not None:

                chunk = chunk[
                    chunk["peak_frp"]
                    <= max_frp
                ]

            # ------------------------------------------------
            # ANOMALY
            # ------------------------------------------------

            if anomaly is not None:

                values = pd.to_numeric(
                    chunk["is_anomaly"],
                    errors="coerce"
                ).fillna(0).astype(int)

                chunk = chunk[
                    values
                    == int(anomaly)
                ]

            # ------------------------------------------------
            # PERSISTENT
            # ------------------------------------------------

            if persistent is not None:

                values = pd.to_numeric(
                    chunk["persistent"],
                    errors="coerce"
                ).fillna(0).astype(int)

                chunk = chunk[
                    values
                    == int(persistent)
                ]

            # ------------------------------------------------
            # LATITUDE
            # ------------------------------------------------

            if (
                min_lat is not None
                or max_lat is not None
            ):

                chunk["latitude"] = pd.to_numeric(
                    chunk["latitude"],
                    errors="coerce"
                )

            if min_lat is not None:

                chunk = chunk[
                    chunk["latitude"]
                    >= min_lat
                ]

            if max_lat is not None:

                chunk = chunk[
                    chunk["latitude"]
                    <= max_lat
                ]

            # ------------------------------------------------
            # LONGITUDE
            # ------------------------------------------------

            if (
                min_lon is not None
                or max_lon is not None
            ):

                chunk["longitude"] = pd.to_numeric(
                    chunk["longitude"],
                    errors="coerce"
                )

            if min_lon is not None:

                chunk = chunk[
                    chunk["longitude"]
                    >= min_lon
                ]

            if max_lon is not None:

                chunk = chunk[
                    chunk["longitude"]
                    <= max_lon
                ]

            # ------------------------------------------------
            # EMPTY CHUNK
            # ------------------------------------------------

            if chunk.empty:
                continue

            # ------------------------------------------------
            # SORT EACH CHUNK
            # ------------------------------------------------

            if sort_by == "frp":

                chunk = chunk.sort_values(
                    "peak_frp",
                    ascending=False
                )

            else:

                chunk["_sort_date"] = (
                    chunk["event_date"]
                    .astype(str)
                )

                chunk = chunk.sort_values(
                    "_sort_date",
                    ascending=False
                )

            # ------------------------------------------------
            # KEEP ONLY REQUIRED NUMBER
            # ------------------------------------------------

            chunk = chunk.head(limit)

            collected.append(
                chunk
            )

            # ------------------------------------------------
            # MEMORY PROTECTION
            # ------------------------------------------------

            if len(collected) >= 10:

                combined = pd.concat(
                    collected,
                    ignore_index=True
                )

                if sort_by == "frp":

                    combined = (
                        combined
                        .sort_values(
                            "peak_frp",
                            ascending=False
                        )
                        .head(limit)
                    )

                else:

                    combined = (
                        combined
                        .sort_values(
                            "_sort_date",
                            ascending=False
                        )
                        .head(limit)
                    )

                collected = [
                    combined
                ]

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not collected:

            return {
                "status": "success",
                "count": 0,
                "events": [],
            }

        # ----------------------------------------------------
        # COMBINE SMALL RESULTS
        # ----------------------------------------------------

        result_df = pd.concat(
            collected,
            ignore_index=True
        )

        # ----------------------------------------------------
        # FINAL SORT
        # ----------------------------------------------------

        if sort_by == "frp":

            result_df["peak_frp"] = pd.to_numeric(
                result_df["peak_frp"],
                errors="coerce"
            ).fillna(0)

            result_df = (
                result_df
                .sort_values(
                    "peak_frp",
                    ascending=False
                )
                .head(limit)
            )

        else:

            result_df["_sort_date"] = (
                result_df["event_date"]
                .astype(str)
            )

            result_df = (
                result_df
                .sort_values(
                    "_sort_date",
                    ascending=False
                )
                .head(limit)
            )

        # ----------------------------------------------------
        # SERIALIZE
        # ----------------------------------------------------

        result = [
            serialize_event(row)
            for _, row in result_df.iterrows()
        ]

        return {
            "status": "success",
            "count": len(result),
            "events": result,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset not found: {exc}"
            ),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid event query: {exc}"
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not retrieve events: {exc}"
            ),
        )


# ============================================================
# SINGLE EVENT PREDICTION
# ============================================================

@app.post("/predict-event")
def predict_single_event(
    request: EventRequest
):

    try:

        df = load_event_data()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    event_df = df[
        df["event_id"]
        == request.event_id
    ]

    if event_df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Event ID {request.event_id} "
                "not found"
            )
        )

    row = event_df.iloc[0]

    features = get_feature_dict(
        row
    )

    try:

        prediction = predict_event(
            features
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"ML prediction failed: {exc}"
            )
        )

    latitude = safe_float(
        row.get("latitude")
    )

    longitude = safe_float(
        row.get("longitude")
    )

    return {

        "status": "success",

        "event_id": safe_int(
            row.get("event_id")
        ),

        "event": {

            "latitude": latitude,

            "longitude": longitude,

            "event_date": str(
                row.get(
                    "event_date",
                    ""
                )
            ),

            "start_time": str(
                row.get(
                    "start_time",
                    ""
                )
            ),

            "end_time": str(
                row.get(
                    "end_time",
                    ""
                )
            ),

            "observation_count": safe_int(
                row.get(
                    "observation_count"
                )
            ),

            "mean_frp": safe_float(
                row.get(
                    "mean_frp"
                )
            ),

            "peak_frp": safe_float(
                row.get(
                    "peak_frp"
                )
            ),

            "total_frp": safe_float(
                row.get(
                    "total_frp"
                )
            ),
        },

        "location": {

            "latitude": latitude,

            "longitude": longitude,

            "type": "Satellite hotspot",

            "maps": create_map_links(
                latitude,
                longitude
            ),
        },

        "prediction": {

            "predicted_source": (
                prediction[
                    "predicted_source"
                ]
            ),

            "source": (
                prediction[
                    "predicted_source"
                ]
            ),

            "confidence": (
                prediction[
                    "confidence"
                ]
            ),

            "probabilities": (
                prediction[
                    "probabilities"
                ]
            ),
        },

        "probabilities": (
            prediction[
                "probabilities"
            ]
        ),

        "anomaly": {

            "is_anomaly": bool(
                safe_int(
                    row.get(
                        "is_anomaly"
                    )
                )
            ),

            "anomaly_score": safe_float(
                row.get(
                    "anomaly_score"
                )
            ),
        },
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict-batch")
def predict_batch(
    request: BatchEventRequest
):

    try:

        df = load_event_data()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    event_df = df[
        df["event_id"].isin(
            request.event_ids
        )
    ]

    results = []

    found_ids = set()

    for _, row in event_df.iterrows():

        event_id = safe_int(
            row.get("event_id")
        )

        found_ids.add(
            event_id
        )

        try:

            prediction = predict_event(
                get_feature_dict(row)
            )

        except Exception:

            continue

        latitude = safe_float(
            row.get("latitude")
        )

        longitude = safe_float(
            row.get("longitude")
        )

        results.append({

            "event_id": event_id,

            "latitude": latitude,

            "longitude": longitude,

            "event_date": str(
                row.get(
                    "event_date",
                    ""
                )
            ),

            "mean_frp": safe_float(
                row.get(
                    "mean_frp"
                )
            ),

            "peak_frp": safe_float(
                row.get(
                    "peak_frp"
                )
            ),

            "source": (
                prediction[
                    "predicted_source"
                ]
            ),

            "confidence": (
                prediction[
                    "confidence"
                ]
            ),

            "probabilities": (
                prediction[
                    "probabilities"
                ]
            ),

            "anomaly": bool(
                safe_int(
                    row.get(
                        "is_anomaly"
                    )
                )
            ),

            "anomaly_score": safe_float(
                row.get(
                    "anomaly_score"
                )
            ),

            "maps": create_map_links(
                latitude,
                longitude
            ),
        })

    missing = [

        event_id

        for event_id in request.event_ids

        if event_id not in found_ids

    ]

    return {

        "status": "success",

        "requested_count": len(
            request.event_ids
        ),

        "processed_count": len(
            results
        ),

        "missing_event_ids": missing,

        "results": results,
    }


# ============================================================
# EVENT ANALYSIS
# ============================================================

@app.get(
    "/event-analysis/{event_id}"
)
def event_analysis(
    event_id: int
):

    try:

        df = load_event_data()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    event_df = df[
        df["event_id"]
        == event_id
    ]

    if event_df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Event {event_id} "
                "not found"
            )
        )

    row = event_df.iloc[0]

    try:

        prediction = predict_event(
            get_feature_dict(row)
        )

    except Exception as exc:

        prediction = {

            "predicted_source":
                "Uncertain",

            "confidence": 0.0,

            "probabilities": {},

            "error": str(exc),
        }

    latitude = safe_float(
        row.get("latitude")
    )

    longitude = safe_float(
        row.get("longitude")
    )

    sentinel = {

        "available": False,

        "source": None,
    }

    if get_sentinel_status is not None:

        try:

            sentinel = (
                get_sentinel_status(
                    event_id
                )
            )

        except Exception:

            pass

    return {

        "status": "success",

        "event_id": event_id,

        "event": serialize_event(
            row
        ),

        "location": {

            "latitude": latitude,

            "longitude": longitude,

            "maps": create_map_links(
                latitude,
                longitude
            ),
        },

        "ml_prediction": {

            "predicted_source": (
                prediction[
                    "predicted_source"
                ]
            ),

            "confidence": (
                prediction[
                    "confidence"
                ]
            ),

            "probabilities": (
                prediction[
                    "probabilities"
                ]
            ),
        },

        "anomaly": {

            "is_anomaly": bool(
                safe_int(
                    row.get(
                        "is_anomaly"
                    )
                )
            ),

            "anomaly_score": (
                safe_float(
                    row.get(
                        "anomaly_score"
                    )
                )
            ),
        },

        "sentinel": sentinel,
    }


# ============================================================
# SENTINEL STATUS
# ============================================================

@app.get(
    "/sentinel/{event_id}"
)
def sentinel_status(
    event_id: int
):

    if get_sentinel_status is None:

        return {

            "status": "success",

            "event_id": event_id,

            "available": False,

            "source": None,

            "message": (
                "Sentinel manager "
                "is not available."
            ),
        }

    try:

        result = get_sentinel_status(
            event_id
        )

        return {

            "status": "success",

            "event_id": event_id,

            **result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# SENTINEL FILES
# ============================================================

@app.get(
    "/sentinel/{event_id}/files"
)
def sentinel_files(
    event_id: int
):

    if get_sentinel_files is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Sentinel manager "
                "unavailable"
            )
        )

    try:

        files = get_sentinel_files(
            event_id
        )

        return {

            "status": "success",

            "event_id": event_id,

            "files": files,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# SENTINEL EVIDENCE
# ============================================================

@app.get(
    "/sentinel/{event_id}/evidence"
)
def sentinel_evidence(
    event_id: int
):

    if calculate_sentinel_indices is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Sentinel processing "
                "unavailable"
            )
        )

    try:

        df = load_event_data()

        event_df = df[
            df["event_id"]
            == event_id
        ]

        if event_df.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Event {event_id} "
                    "not found"
                )
            )

        row = event_df.iloc[0]

        latitude = safe_float(
            row.get("latitude")
        )

        longitude = safe_float(
            row.get("longitude")
        )

        evidence = (
            calculate_sentinel_indices(
                latitude=latitude,
                longitude=longitude,
                event_id=event_id,
                buffer_m=500,
            )
        )

        return {

            "status": "success",

            "event_id": event_id,

            "location": {

                "latitude": latitude,

                "longitude": longitude,
            },

            "sentinel": {

                "available": True,

                "source": "local_cache",
            },

            "evidence": evidence,
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# SENTINEL VISUALIZATIONS
# ============================================================

VISUALIZATION_TYPES = {

    "ndvi": "NDVI",

    "nbr": "NBR",

    "scl": "SCL",

    "false_color": "false_color",
}


def find_visualization(
    event_id: int,
    visualization_type: str
):

    event_dir = (

        BASE_DIR
        / "data"
        / "sentinel"
        / f"event_{event_id}"
        / "visualizations"

    )

    if not event_dir.exists():

        return None

    name = VISUALIZATION_TYPES.get(
        visualization_type,
        visualization_type
    )

    candidates = list(
        event_dir.glob(
            f"*_{name}.png"
        )
    )

    if not candidates:

        candidates = list(
            event_dir.glob(
                f"*{name}*.png"
            )
        )

    if not candidates:

        return None

    return candidates[0]


@app.get(
    "/sentinel/{event_id}/visualizations"
)
def sentinel_visualizations(
    event_id: int
):

    result = {}

    for visualization_type in (
        VISUALIZATION_TYPES
    ):

        path = find_visualization(
            event_id,
            visualization_type
        )

        result[
            visualization_type
        ] = path is not None

    return {

        "status": "success",

        "event_id": event_id,

        "visualizations": result,
    }


@app.get(
    "/sentinel/{event_id}/visualizations/{visualization_type}"
)
def sentinel_visualization(
    event_id: int,
    visualization_type: str
):

    if (
        visualization_type
        not in VISUALIZATION_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Visualization must be "
                "one of: ndvi, nbr, "
                "scl, false_color"
            ),
        )

    path = find_visualization(
        event_id,
        visualization_type
    )

    if path is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"{visualization_type} "
                f"visualization not found "
                f"for event {event_id}"
            ),
        )

    return FileResponse(
        path=str(path),
        media_type="image/png",
        filename=path.name,
    )


# ============================================================
# SERVER INFO
# ============================================================

@app.get(
    "/server-info"
)
def server_info():

    try:

        df = load_event_data()

        event_count = len(df)

    except Exception:

        event_count = 0

    return {

        "service": "VEYRONIX AI",

        "problem_statement": "SIH26162",

        "version": "2.0.0",

        "authentication":
            "disabled_for_local_SIH_demo",

        "dataset": {

            "available":
                DATA_FILE.exists(),

            "events":
                event_count,
        },

        "model": {

            "name": "LightGBM",

            "available":
                callable(
                    predict_event
                ),
        },

        "sentinel": {

            "enabled":
                calculate_sentinel_indices
                is not None,
        },
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

