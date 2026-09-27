from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# VEYRONIX AI API
# SIH 2026 - SIH26162
# ============================================================

APP_NAME = "VEYRONIX AI"
APP_VERSION = "2.2.0"

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lightgbm_source_classifier.pkl"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "AI-based detection and classification of industrial fires "
        "and persistent thermal sources using NASA FIRMS and satellite data."
    ),
    version=APP_VERSION,
)


# ============================================================
# CORS
# ============================================================
#
# IMPORTANT:
# allow_origins=["*"] is intentionally used for the current
# SIH prototype so that:
#
# localhost React
# Netlify frontend
# Streamlit
# other demo clients
#
# can communicate with the Render API.
#
# allow_credentials must remain False when using "*".
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL CACHE
# ============================================================

MODEL: Any = None
DATASET_AVAILABLE = False
MODEL_AVAILABLE = False

EVENT_CACHE: pd.DataFrame | None = None

FEATURE_NAMES = [
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

EVENT_COLUMNS = [
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


# ============================================================
# PYDANTIC SCHEMAS
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
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.lower().strip() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return False


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(minimum, min(maximum, value))


def json_safe(value: Any) -> Any:

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:

    records = df.to_dict(orient="records")

    cleaned = []

    for record in records:

        cleaned_record = {
            key: json_safe(value)
            for key, value in record.items()
        }

        cleaned.append(cleaned_record)

    return cleaned


# ============================================================
# MODEL LOADING
# ============================================================

def load_model() -> None:

    global MODEL
    global MODEL_AVAILABLE

    MODEL_AVAILABLE = False
    MODEL = None

    if not MODEL_PATH.exists():

        print(
            f"[VEYRONIX] Model not found: {MODEL_PATH}"
        )

        return

    try:

        MODEL = joblib.load(MODEL_PATH)

        MODEL_AVAILABLE = True

        print(
            "[VEYRONIX] LightGBM model loaded successfully."
        )

        print(
            f"[VEYRONIX] Model type: {type(MODEL)}"
        )

    except Exception as exc:

        print(
            "[VEYRONIX] Model loading failed:"
        )

        print(exc)


# ============================================================
# DATASET CACHE
# ============================================================

def load_latest_events_cache() -> None:

    global EVENT_CACHE
    global DATASET_AVAILABLE

    DATASET_AVAILABLE = False
    EVENT_CACHE = None

    if not DATASET_PATH.exists():

        print(
            f"[VEYRONIX] Dataset not found: {DATASET_PATH}"
        )

        return

    try:

        print(
            "[VEYRONIX] Loading event cache..."
        )

        df = pd.read_csv(
            DATASET_PATH,
            usecols=lambda column: (
                column in EVENT_COLUMNS
            ),
        )

        # Ensure expected columns exist.
        for column in EVENT_COLUMNS:

            if column not in df.columns:

                df[column] = np.nan

        # Event ID
        df["event_id"] = pd.to_numeric(
            df["event_id"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["event_id"]
        )

        df["event_id"] = (
            df["event_id"]
            .astype(int)
        )

        # Dates
        if "event_date" in df.columns:

            df["event_date"] = pd.to_datetime(
                df["event_date"],
                errors="coerce",
            )

        if "start_time" in df.columns:

            df["start_time"] = pd.to_datetime(
                df["start_time"],
                errors="coerce",
            )

        # Sort chronologically.
        sort_columns = []

        if "event_date" in df.columns:
            sort_columns.append("event_date")

        if "start_time" in df.columns:
            sort_columns.append("start_time")

        if sort_columns:

            df = df.sort_values(
                sort_columns,
                ascending=True,
            )

        # Keep unique events.
        df = df.drop_duplicates(
            subset=["event_id"],
            keep="last",
        )

        # Keep newest 1000 events in memory.
        df = df.tail(1000)

        EVENT_CACHE = df.reset_index(
            drop=True
        )

        DATASET_AVAILABLE = True

        print(
            f"[VEYRONIX] Event cache loaded: "
            f"{len(EVENT_CACHE)} events"
        )

    except Exception as exc:

        print(
            "[VEYRONIX] Dataset cache loading failed:"
        )

        print(exc)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "[VEYRONIX] Starting VEYRONIX AI API..."
    )

    load_model()

    load_latest_events_cache()

    print(
        "[VEYRONIX] Startup complete."
    )


# ============================================================
# EVENT RECORD HELPERS
# ============================================================

def get_event_from_cache(
    event_id: int,
) -> dict[str, Any] | None:

    if EVENT_CACHE is None:
        return None

    matches = EVENT_CACHE[
        EVENT_CACHE["event_id"] == event_id
    ]

    if matches.empty:
        return None

    record = matches.iloc[-1].to_dict()

    return {
        key: json_safe(value)
        for key, value in record.items()
    }


def load_event_from_dataset(
    event_id: int,
) -> dict[str, Any] | None:

    if not DATASET_PATH.exists():
        return None

    try:

        for chunk in pd.read_csv(
            DATASET_PATH,
            chunksize=50000,
        ):

            if "event_id" not in chunk.columns:
                continue

            chunk["event_id"] = pd.to_numeric(
                chunk["event_id"],
                errors="coerce",
            )

            matches = chunk[
                chunk["event_id"] == event_id
            ]

            if not matches.empty:

                record = matches.iloc[0].to_dict()

                return {
                    key: json_safe(value)
                    for key, value in record.items()
                }

    except Exception as exc:

        print(
            "[VEYRONIX] Event lookup failed:"
        )

        print(exc)

    return None


def get_event(
    event_id: int,
) -> dict[str, Any] | None:

    event = get_event_from_cache(
        event_id
    )

    if event is not None:
        return event

    return load_event_from_dataset(
        event_id
    )


# ============================================================
# MODEL INPUT
# ============================================================

def build_model_input(
    event: dict[str, Any],
) -> pd.DataFrame:

    row = {}

    for feature in FEATURE_NAMES:

        row[feature] = safe_float(
            event.get(feature, 0)
        )

    return pd.DataFrame(
        [row],
        columns=FEATURE_NAMES,
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_event_source(
    event: dict[str, Any],
) -> dict[str, Any]:

    if not MODEL_AVAILABLE or MODEL is None:

        return {
            "predicted_source": "Uncertain",
            "confidence": 0.0,
            "probabilities": {},
            "model_available": False,
        }

    try:

        X = build_model_input(
            event
        )

        probabilities_array = MODEL.predict_proba(
            X
        )[0]

        classes = MODEL.classes_

        probabilities = {}

        class_names = {
            0: "Agriculture_Biomass",
            1: "Forest_Natural",
            2: "Industrial",
            3: "Waste_Other",
        }

        for class_id, probability in zip(
            classes,
            probabilities_array,
        ):

            class_id_int = int(class_id)

            class_name = class_names.get(
                class_id_int,
                str(class_id_int),
            )

            probabilities[class_name] = round(
                float(probability),
                4,
            )

        best_index = int(
            np.argmax(
                probabilities_array
            )
        )

        best_class_id = int(
            classes[best_index]
        )

        predicted_source = class_names.get(
            best_class_id,
            "Uncertain",
        )

        confidence = float(
            probabilities_array[
                best_index
            ]
        )

        return {
            "predicted_source": predicted_source,
            "confidence": round(
                confidence,
                4,
            ),
            "probabilities": probabilities,
            "model_available": True,
        }

    except Exception as exc:

        print(
            "[VEYRONIX] Prediction failed:"
        )

        print(exc)

        return {
            "predicted_source": "Uncertain",
            "confidence": 0.0,
            "probabilities": {},
            "model_available": True,
            "error": str(exc),
        }


# ============================================================
# PRIORITY
# ============================================================

def calculate_priority(
    confidence: float,
    anomaly: bool,
    peak_frp: float,
) -> tuple[str, int]:

    score = (
        confidence * 60
        + (30 if anomaly else 0)
        + min(10, peak_frp / 10)
    )

    score = int(
        round(
            clamp(
                score,
                0,
                100,
            )
        )
    )

    if score >= 85:
        priority = "CRITICAL"

    elif score >= 70:
        priority = "HIGH"

    elif score >= 50:
        priority = "MEDIUM"

    else:
        priority = "LOW"

    return priority, score


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "events": "/events",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_available": MODEL_AVAILABLE,
        "dataset_available": DATASET_AVAILABLE,
        "api_key_configured": bool(
            os.getenv("VEYRONIX_API_KEY")
        ),
        "authentication": (
            "enabled"
            if os.getenv("VEYRONIX_API_KEY")
            else "disabled_for_local_SIH_demo"
        ),
        "cors_enabled": True,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    model_type = (
        str(type(MODEL))
        if MODEL is not None
        else None
    )

    return {
        "status": "success",
        "model_available": MODEL_AVAILABLE,
        "model_path": str(MODEL_PATH),
        "model_type": model_type,
        "features": FEATURE_NAMES,
        "classes": [
            "Agriculture_Biomass",
            "Forest_Natural",
            "Industrial",
            "Waste_Other",
        ],
    }


# ============================================================
# EVENTS
# ============================================================

@app.get("/events")
def events(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    start_date: str | None = None,
    end_date: str | None = None,
):

    if not DATASET_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail="Event dataset is unavailable.",
        )

    # --------------------------------------------------------
    # Fast path
    # --------------------------------------------------------

    if (
        start_date is None
        and end_date is None
        and EVENT_CACHE is not None
    ):

        result = EVENT_CACHE.tail(
            limit
        ).copy()

    else:

        try:

            frames = []

            for chunk in pd.read_csv(
                DATASET_PATH,
                chunksize=50000,
            ):

                if "event_id" not in chunk.columns:
                    continue

                if (
                    start_date is not None
                    or end_date is not None
                ):

                    chunk["event_date"] = pd.to_datetime(
                        chunk["event_date"],
                        errors="coerce",
                    )

                if start_date is not None:

                    start = pd.Timestamp(
                        start_date
                    )

                    chunk = chunk[
                        chunk["event_date"]
                        >= start
                    ]

                if end_date is not None:

                    end = pd.Timestamp(
                        end_date
                    )

                    chunk = chunk[
                        chunk["event_date"]
                        <= end
                    ]

                if not chunk.empty:

                    frames.append(chunk)

            if frames:

                result = pd.concat(
                    frames,
                    ignore_index=True,
                )

                result = result.drop_duplicates(
                    subset=["event_id"],
                    keep="last",
                )

                result = result.tail(
                    limit
                )

            else:

                result = pd.DataFrame()

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to read event dataset: "
                    f"{exc}"
                ),
            )

    if result.empty:

        return {
            "status": "success",
            "count": 0,
            "events": [],
        }

    # --------------------------------------------------------
    # Build frontend-friendly response
    # --------------------------------------------------------

    output = []

    for _, row in result.iterrows():

        event_id = safe_int(
            row.get("event_id")
        )

        latitude = safe_float(
            row.get("latitude")
        )

        longitude = safe_float(
            row.get("longitude")
        )

        event_date = row.get(
            "event_date"
        )

        start_time = row.get(
            "start_time"
        )

        end_time = row.get(
            "end_time"
        )

        event_date_string = (
            str(event_date.date())
            if isinstance(
                event_date,
                pd.Timestamp,
            )
            else str(event_date)
        )

        start_time_string = (
            str(start_time)
            if not pd.isna(start_time)
            else None
        )

        end_time_string = (
            str(end_time)
            if not pd.isna(end_time)
            else None
        )

        output.append(
            {
                "event_id": event_id,
                "latitude": latitude,
                "longitude": longitude,
                "event_date": event_date_string,
                "start_time": start_time_string,
                "end_time": end_time_string,

                "observation_count": safe_float(
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
                    row.get(
                        "historical_detection_count"
                    )
                ),

                "historical_mean_frp": safe_float(
                    row.get(
                        "historical_mean_frp"
                    )
                ),

                "local_frp_deviation": safe_float(
                    row.get(
                        "local_frp_deviation"
                    )
                ),

                "historical_daily_activity": safe_float(
                    row.get(
                        "historical_daily_activity"
                    )
                ),

                "previously_detected": safe_float(
                    row.get(
                        "previously_detected"
                    )
                ),

                "anomaly_score": safe_float(
                    row.get(
                        "anomaly_score"
                    )
                ),

                "is_anomaly": safe_bool(
                    row.get(
                        "is_anomaly"
                    )
                ),

                "maps": {
                    "google_maps": (
                        "https://www.google.com/maps/search/"
                        f"?api=1&query={latitude},{longitude}"
                    ),
                    "openstreetmap": (
                        "https://www.openstreetmap.org/"
                        f"?mlat={latitude}"
                        f"&mlon={longitude}"
                        "&zoom=15"
                    ),
                },
            }
        )

    return {
        "status": "success",
        "count": len(output),
        "events": output,
    }


# ============================================================
# PREDICT EVENT
# ============================================================

@app.post("/predict-event")
def predict_event(
    request: EventRequest,
):

    event_id = request.event_id

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    prediction = predict_event_source(
        event
    )

    return {
        "status": "success",
        "event_id": event_id,
        **prediction,
    }


# ============================================================
# PREDICT BATCH
# ============================================================

@app.post("/predict-batch")
def predict_batch(
    request: BatchEventRequest,
):

    if len(request.event_ids) > 100:

        raise HTTPException(
            status_code=400,
            detail="Maximum 100 events per batch.",
        )

    predictions = []

    for event_id in request.event_ids:

        event = get_event(
            event_id
        )

        if event is None:

            predictions.append(
                {
                    "event_id": event_id,
                    "status": "not_found",
                }
            )

            continue

        prediction = predict_event_source(
            event
        )

        predictions.append(
            {
                "event_id": event_id,
                "status": "success",
                **prediction,
            }
        )

    return {
        "status": "success",
        "count": len(predictions),
        "predictions": predictions,
    }


# ============================================================
# EVENT ANALYSIS
# ============================================================

@app.get("/event-analysis/{event_id}")
def event_analysis(
    event_id: int,
):

    started = time.perf_counter()

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    prediction = predict_event_source(
        event
    )

    predicted_source = prediction.get(
        "predicted_source",
        "Uncertain",
    )

    confidence = safe_float(
        prediction.get(
            "confidence",
            0,
        )
    )

    probabilities = prediction.get(
        "probabilities",
        {},
    )

    anomaly = safe_bool(
        event.get(
            "is_anomaly",
            False,
        )
    )

    anomaly_score = safe_float(
        event.get(
            "anomaly_score",
            0,
        )
    )

    peak_frp = safe_float(
        event.get(
            "peak_frp",
            0,
        )
    )

    priority, priority_score = calculate_priority(
        confidence,
        anomaly,
        peak_frp,
    )

    latitude = safe_float(
        event.get(
            "latitude"
        )
    )

    longitude = safe_float(
        event.get(
            "longitude"
        )
    )

    insufficient_evidence = (
        confidence < 0.55
    )

    if insufficient_evidence:

        source = "Uncertain"

        recommendation = (
            "Additional satellite and contextual evidence "
            "is required before source attribution."
        )

        message = (
            "Evidence is insufficient for reliable source attribution."
        )

    else:

        source = predicted_source

        recommendation = (
            "Review satellite evidence, OSM context, "
            "historical behaviour and investigator observations."
        )

        message = (
            f"VEYRONIX predicts {predicted_source} "
            f"with {confidence * 100:.1f}% model confidence."
        )

    evidence = [
        (
            "NASA FIRMS thermal event detected at "
            f"{latitude:.4f}, {longitude:.4f}"
        ),
        (
            f"Peak FRP: {peak_frp:.1f} MW"
        ),
        (
            "Observation count: "
            f"{safe_float(event.get('observation_count')):.0f}"
        ),
        (
            "Mean brightness: "
            f"{safe_float(event.get('mean_brightness')):.1f} K"
        ),
        (
            "Persistent event: "
            f"{'Yes' if safe_bool(event.get('persistent')) else 'No'}"
        ),
        (
            "Anomaly status: "
            f"{'Anomalous' if anomaly else 'Not anomalous'}"
        ),
    ]

    if source != "Uncertain":

        evidence.append(
            f"VEYRONIX ML prediction: {source}"
        )

    else:

        evidence.append(
            "Source attribution remains uncertain."
        )

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000

    return {
        "status": "success",

        "event_id": event_id,

        "latitude": latitude,
        "longitude": longitude,

        "event": event,

        "source": source,
        "predicted_source": predicted_source,

        "confidence": confidence,

        "source_probabilities": probabilities,

        "abnormality": (
            "HIGH"
            if anomaly
            else "LOW"
        ),

        "abnormality_score": round(
            anomaly_score * 100,
            2,
        ),

        "priority": priority,
        "priority_score": priority_score,

        "evidence": evidence,

        "insufficient_evidence": (
            insufficient_evidence
        ),

        "message": message,

        "recommendation": recommendation,

        "anomaly": {
            "is_anomaly": anomaly,
            "anomaly_score": anomaly_score,
        },

        "sentinel": {
            "status": "pending",
            "message": (
                "Sentinel evidence can be requested "
                "for this event."
            ),
        },

        "analysis_timestamp": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),

        "processing_latency_ms": round(
            elapsed_ms,
            2,
        ),
    }


# ============================================================
# SENTINEL ENDPOINTS
# ============================================================

@app.get("/sentinel/{event_id}")
def sentinel_event(
    event_id: int,
):

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    return {
        "status": "success",
        "event_id": event_id,
        "sentinel_available": False,
        "message": (
            "Sentinel evidence integration is available "
            "through the VEYRONIX satellite evidence pipeline."
        ),
    }


@app.get("/sentinel/{event_id}/files")
def sentinel_files(
    event_id: int,
):

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    return {
        "status": "success",
        "event_id": event_id,
        "files": [],
        "message": (
            "No Sentinel files are currently attached "
            "to this event."
        ),
    }


@app.get("/sentinel/{event_id}/evidence")
def sentinel_evidence(
    event_id: int,
):

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    return {
        "status": "success",
        "event_id": event_id,
        "available": False,
        "evidence": [],
        "message": (
            "High-resolution Sentinel evidence "
            "is not currently attached to this event."
        ),
    }


@app.get("/sentinel/{event_id}/visualizations")
def sentinel_visualizations(
    event_id: int,
):

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    return {
        "status": "success",
        "event_id": event_id,
        "visualizations": [],
        "message": (
            "No Sentinel visualizations are currently "
            "attached to this event."
        ),
    }


@app.get(
    "/sentinel/{event_id}/visualizations/{visualization_type}"
)
def sentinel_visualization(
    event_id: int,
    visualization_type: str,
):

    event = get_event(
        event_id
    )

    if event is None:

        raise HTTPException(
            status_code=404,
            detail=f"Event {event_id} not found.",
        )

    return {
        "status": "success",
        "event_id": event_id,
        "visualization_type": visualization_type,
        "available": False,
        "message": (
            "Requested Sentinel visualization "
            "is not currently attached."
        ),
    }


# ============================================================
# SERVER INFO
# ============================================================

@app.get("/server-info")
def server_info():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",

        "model_available": MODEL_AVAILABLE,
        "dataset_available": DATASET_AVAILABLE,

        "model_path": str(
            MODEL_PATH
        ),

        "dataset_path": str(
            DATASET_PATH
        ),

        "cached_events": (
            len(EVENT_CACHE)
            if EVENT_CACHE is not None
            else 0
        ),

        "cors": {
            "enabled": True,
            "allow_origins": ["*"],
            "allow_credentials": False,
        },

        "endpoints": [
            "/",
            "/health",
            "/model-info",
            "/events",
            "/predict",
            "/predict-event",
            "/predict-batch",
            "/event-analysis/{event_id}",
            "/sentinel/{event_id}",
            "/sentinel/{event_id}/files",
            "/sentinel/{event_id}/evidence",
            "/sentinel/{event_id}/visualizations",
            "/server-info",
        ],
    }


# ============================================================
# GENERIC PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    event: ThermalEvent,
):

    event_dict = event.model_dump()

    prediction = predict_event_source(
        event_dict
    )

    return {
        "status": "success",
        **prediction,
    }