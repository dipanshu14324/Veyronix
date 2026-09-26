# ============================================================
# VEYRONIX AI
# SIH26162
# Industrial Fire Detection & Source Attribution
#
# Dashboard:
#   Streamlit
#   FastAPI
#   LightGBM
#   NASA FIRMS
#   OSM
#   Sentinel-2
# ============================================================

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VEYRONIX AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ML_DATA = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)

OSM_DATA = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "supervised_source_candidates.csv"
)

OSM_GPKG_CANDIDATES = [
    BASE_DIR
    / "data"
    / "raw_data"
    / "western-zone-260923-free.gpkg",

    BASE_DIR
    / "data"
    / "raw_data"
    / "western-zone.gpkg",
]


# ============================================================
# API CONFIGURATION
# ============================================================

# IMPORTANT:
# Streamlit Cloud should use the Render FastAPI backend.
#
# If API_URL exists in Streamlit Secrets/environment,
# it will be used automatically.
#
# Otherwise this Render URL is used.
API_URL = os.getenv(
    "API_URL",
    "https://veyronix.onrender.com",
).rstrip("/")


VEYRONIX_API_KEY = os.getenv(
    "VEYRONIX_API_KEY",
    "",
).strip()


API_HEADERS = {}

if VEYRONIX_API_KEY:
    API_HEADERS["X-API-Key"] = VEYRONIX_API_KEY


# ============================================================
# REQUEST CONFIGURATION
# ============================================================

API_CONNECT_TIMEOUT = 10
API_READ_TIMEOUT = 60

# Render Free services can sleep.
# Multiple attempts give Render time to wake.
API_RETRIES = 3

API_RETRY_DELAYS = [
    2,
    8,
    15,
]


# ============================================================
# SESSION STATE
# ============================================================

if "api_last_status" not in st.session_state:
    st.session_state["api_last_status"] = "Not tested"

if "api_last_error" not in st.session_state:
    st.session_state["api_last_error"] = ""

if "api_last_endpoint" not in st.session_state:
    st.session_state["api_last_endpoint"] = ""

if "selected_event" not in st.session_state:
    st.session_state["selected_event"] = 84118


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #05070b;
        color: #f8fafc;
    }

    .main .block-container {
        background: #05070b;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }

    p {
        color: #aeb8c7;
    }

    label {
        color: #d8dee9 !important;
    }

    .stCaption {
        color: #7f8a9a !important;
    }

    section[data-testid="stSidebar"] {
        background: #080c14;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #273244;
    }

    div[data-testid="stMetric"] {
        background: #0b111c;
        border: 1px solid #1d2939;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }

    div[data-testid="stMetricLabel"] {
        color: #8995a7 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
    }

    .stButton > button {
        background: #111827;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #1f2937;
        color: #ffffff;
        border-color: #ef4444;
    }

    .stButton > button[kind="primary"] {
        background: #dc2626;
        color: white;
        border: 1px solid #ef4444;
    }

    .stButton > button[kind="primary"]:hover {
        background: #b91c1c;
        border-color: #f87171;
    }

    div[data-baseweb="input"] {
        background: #0b111c;
        border-color: #334155;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
        background: #0b111c !important;
    }

    div[data-baseweb="select"] > div {
        background: #0b111c;
        border-color: #334155;
        color: #f8fafc;
    }

    div[data-testid="stNumberInput"] input {
        background: #0b111c !important;
        color: #f8fafc !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1f2937;
        background: #0b111c;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #1f2937;
        border-radius: 12px;
        background: #0b111c;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #1f2937 !important;
    }

    iframe {
        border-radius: 12px;
        border: 1px solid #1f2937;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_bool(value):

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, str):

        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def normalize_source(source):

    if source is None:
        return "Uncertain"

    source = str(source).strip()

    mapping = {

        "Industrial":
            "Industrial",

        "industrial":
            "Industrial",

        "Agriculture_Biomass":
            "Agriculture / Biomass",

        "Agriculture":
            "Agriculture / Biomass",

        "agriculture":
            "Agriculture / Biomass",

        "Forest_Natural":
            "Forest / Natural",

        "Forest":
            "Forest / Natural",

        "forest":
            "Forest / Natural",

        "Waste_Other":
            "Waste / Other",

        "Waste":
            "Waste / Other",

        "waste":
            "Waste / Other",

        "Unknown":
            "Uncertain",

        "unknown":
            "Uncertain",
    }

    return mapping.get(
        source,
        source,
    )


def source_icon(source):

    source = normalize_source(source)

    return {

        "Industrial":
            "🏭",

        "Agriculture / Biomass":
            "🌾",

        "Forest / Natural":
            "🌲",

        "Waste / Other":
            "♻️",

        "Uncertain":
            "❓",

    }.get(
        source,
        "🔥",
    )


def priority_label(row):

    anomaly = safe_bool(
        row.get(
            "is_anomaly",
            False,
        )
    )

    persistent = safe_bool(
        row.get(
            "persistent",
            False,
        )
    )

    try:

        frp = float(
            row.get(
                "peak_frp",
                0,
            )
            or 0
        )

    except Exception:

        frp = 0

    if anomaly and frp >= 20:
        return "CRITICAL"

    if anomaly or frp >= 15:
        return "HIGH"

    if persistent or frp >= 8:
        return "MEDIUM"

    return "LOW"


def format_number(
    value,
    decimals=2,
):

    try:

        if value is None:
            return "N/A"

        return f"{float(value):.{decimals}f}"

    except Exception:

        return "N/A"


# ============================================================
# EVIDENCE RECONCILIATION
# ============================================================

def reconcile_evidence(
    ml_source,
    confidence,
    osm_sources,
):

    ml_source = normalize_source(
        ml_source
    )

    normalized_osm = [
        normalize_source(x)
        for x in osm_sources
        if x is not None
    ]

    normalized_osm = list(
        dict.fromkeys(
            normalized_osm
        )
    )

    if (
        ml_source != "Uncertain"
        and normalized_osm
        and ml_source in normalized_osm
    ):

        return {
            "status":
                "MODEL SUPPORTED",

            "icon":
                "✅",

            "level":
                "success",

            "message":
                (
                    f"VEYRONIX ML predicts "
                    f"**{ml_source} "
                    f"({confidence * 100:.2f}%)**. "
                    f"OSM contextual evidence also "
                    f"indicates **{', '.join(normalized_osm)}**."
                ),

            "osm_sources":
                normalized_osm,
        }

    if (
        ml_source != "Uncertain"
        and normalized_osm
        and ml_source not in normalized_osm
    ):

        return {
            "status":
                "NEEDS REVIEW",

            "icon":
                "⚠️",

            "level":
                "warning",

            "message":
                (
                    f"Primary VEYRONIX ML prediction: "
                    f"**{ml_source} "
                    f"({confidence * 100:.2f}%)**.\n\n"
                    f"OSM contextual land use: "
                    f"**{', '.join(normalized_osm)}**.\n\n"
                    "The supporting evidence differs "
                    "from the ML prediction. Review "
                    "Sentinel-2 and other available "
                    "contextual evidence before "
                    "confirming source attribution."
                ),

            "osm_sources":
                normalized_osm,
        }

    if confidence < 0.60:

        return {
            "status":
                "LOW CONFIDENCE",

            "icon":
                "ℹ️",

            "level":
                "info",

            "message":
                (
                    f"VEYRONIX predicts "
                    f"**{ml_source} "
                    f"({confidence * 100:.2f}%)**, "
                    "but model confidence is low. "
                    "Additional evidence should be "
                    "reviewed before confirmation."
                ),

            "osm_sources":
                normalized_osm,
        }

    return {
        "status":
            "MODEL PREDICTION",

        "icon":
            "ℹ️",

        "level":
            "info",

        "message":
            (
                f"VEYRONIX predicts "
                f"**{ml_source} "
                f"({confidence * 100:.2f}%)**. "
                "No matching OSM contextual evidence "
                "was available."
            ),

        "osm_sources":
            normalized_osm,
    }


# ============================================================
# API REQUEST ENGINE
# ============================================================

def api_request(
    method,
    endpoint,
    retries=API_RETRIES,
    retry_delays=None,
    **kwargs,
):

    if retry_delays is None:
        retry_delays = API_RETRY_DELAYS

    timeout = kwargs.pop(
        "timeout",
        (
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    request_headers = {
        **API_HEADERS,
        **kwargs.pop(
            "headers",
            {},
        ),
    }

    url = f"{API_URL}{endpoint}"

    last_error = None
    last_status = None

    for attempt in range(
        retries
    ):

        try:

            response = requests.request(
                method=method,
                url=url,
                headers=request_headers,
                timeout=timeout,
                **kwargs,
            )

            last_status = response.status_code

            st.session_state[
                "api_last_endpoint"
            ] = endpoint

            st.session_state[
                "api_last_status"
            ] = str(
                response.status_code
            )

            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            if response.ok:

                st.session_state[
                    "api_last_error"
                ] = ""

                return response

            # ------------------------------------------------
            # Render / gateway errors
            # ------------------------------------------------

            if response.status_code in {
                502,
                503,
                504,
            }:

                last_error = (
                    f"HTTP {response.status_code} "
                    f"from Render"
                )

            else:

                try:
                    body = response.text[:500]
                except Exception:
                    body = ""

                last_error = (
                    f"HTTP {response.status_code}"
                    + (
                        f": {body}"
                        if body
                        else ""
                    )
                )

            # Retry gateway failures

            if (
                response.status_code
                not in {
                    502,
                    503,
                    504,
                }
            ):

                break

        except requests.exceptions.Timeout:

            last_error = (
                "Request timed out while "
                "waiting for FastAPI."
            )

        except requests.exceptions.ConnectionError:

            last_error = (
                "Could not connect to "
                "FastAPI/Render."
            )

        except requests.exceptions.RequestException as exc:

            last_error = (
                f"Request error: {exc}"
            )

        except Exception as exc:

            last_error = (
                f"Unexpected API error: {exc}"
            )

        # ----------------------------------------------------
        # Retry delay
        # ----------------------------------------------------

        if attempt < retries - 1:

            if attempt < len(
                retry_delays
            ):

                delay = retry_delays[
                    attempt
                ]

            else:

                delay = 10

            time.sleep(
                delay
            )

    st.session_state[
        "api_last_error"
    ] = (
        last_error
        or "Unknown API error"
    )

    st.session_state[
        "api_last_status"
    ] = (
        str(last_status)
        if last_status
        else "Connection failed"
    )

    return None


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(
    show_spinner="Loading thermal events..."
)
def load_ml_data():

    if not ML_DATA.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            ML_DATA
        )

        if "event_id" in df.columns:

            df["event_id"] = pd.to_numeric(
                df["event_id"],
                errors="coerce",
            )

        if "event_date" in df.columns:

            df["event_date"] = pd.to_datetime(
                df["event_date"],
                errors="coerce",
            )

        return df

    except Exception:

        return pd.DataFrame()


@st.cache_data(
    show_spinner="Loading OSM evidence..."
)
def load_osm_data():

    if not OSM_DATA.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            OSM_DATA
        )

        if "event_id" in df.columns:

            df["event_id"] = pd.to_numeric(
                df["event_id"],
                errors="coerce",
            )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# OSM GEOPACKAGE
# ============================================================

def find_osm_gpkg():

    for path in OSM_GPKG_CANDIDATES:

        if not path.exists():
            continue

        if path.is_file():
            return path

        nested = list(
            path.glob("*.gpkg")
        )

        if nested:
            return nested[0]

    return None


@st.cache_resource(
    show_spinner="Loading OSM land-use layer..."
)
def load_osm_landuse():

    try:

        import geopandas as gpd

    except ImportError:

        return None

    gpkg = find_osm_gpkg()

    if gpkg is None:

        return None

    try:

        layers = gpd.list_layers(
            gpkg
        )

        layer_names = (
            layers["name"]
            .astype(str)
            .tolist()
        )

        target = (
            "gis_osm_landuse_a_free"
        )

        if target not in layer_names:

            possible = [
                x
                for x in layer_names
                if "landuse" in x.lower()
            ]

            if not possible:
                return None

            target = possible[0]

        gdf = gpd.read_file(
            gpkg,
            layer=target,
        )

        if gdf.empty:
            return None

        if gdf.crs is None:

            gdf = gdf.set_crs(
                epsg=4326
            )

        else:

            gdf = gdf.to_crs(
                epsg=4326
            )

        return gdf

    except Exception:

        return None


def osm_source_from_fclass(
    value
):

    if value is None:
        return "Unknown"

    value = str(
        value
    ).strip().lower()

    mapping = {

        "industrial":
            "Industrial",

        "commercial":
            "Industrial",

        "farmland":
            "Agriculture / Biomass",

        "farmyard":
            "Agriculture / Biomass",

        "orchard":
            "Agriculture / Biomass",

        "vineyard":
            "Agriculture / Biomass",

        "forest":
            "Forest / Natural",

        "scrub":
            "Forest / Natural",

        "grass":
            "Forest / Natural",

        "meadow":
            "Forest / Natural",

        "landfill":
            "Waste / Other",

        "quarry":
            "Waste / Other",
    }

    return mapping.get(
        value,
        "Other",
    )


def get_osm_spatial_evidence(
    latitude,
    longitude,
    radius_km=2.0,
):

    gdf = load_osm_landuse()

    if gdf is None:
        return pd.DataFrame()

    try:

        import geopandas as gpd
        from shapely.geometry import Point

        latitude = float(latitude)
        longitude = float(longitude)

        point_wgs84 = Point(
            longitude,
            latitude,
        )

        lat_delta = (
            radius_km
            / 111.0
        )

        lon_delta = (
            radius_km
            /
            max(
                111.0
                * np.cos(
                    np.radians(
                        latitude
                    )
                ),
                1.0,
            )
        )

        bbox = gdf.cx[
            longitude - lon_delta:
            longitude + lon_delta,
            latitude - lat_delta:
            latitude + lat_delta,
        ].copy()

        if bbox.empty:
            return pd.DataFrame()

        bbox = bbox.to_crs(
            epsg=3857
        )

        point = gpd.GeoSeries(
            [point_wgs84],
            crs="EPSG:4326",
        ).to_crs(
            epsg=3857
        ).iloc[0]

        bbox["distance_m"] = (
            bbox.geometry.distance(
                point
            )
        )

        bbox = bbox[
            bbox["distance_m"]
            <= radius_km * 1000
        ].copy()

        if bbox.empty:
            return pd.DataFrame()

        bbox = bbox.sort_values(
            "distance_m"
        ).head(10)

        rows = []

        for _, row in bbox.iterrows():

            fclass = row.get(
                "fclass",
                row.get(
                    "type",
                    row.get(
                        "landuse",
                        "unknown",
                    ),
                ),
            )

            rows.append(
                {
                    "osm_class":
                        str(fclass),

                    "source_context":
                        osm_source_from_fclass(
                            fclass
                        ),

                    "distance_m":
                        round(
                            float(
                                row[
                                    "distance_m"
                                ]
                            ),
                            1,
                        ),
                }
            )

        return pd.DataFrame(
            rows
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# API HEALTH
# ============================================================

@st.cache_data(
    ttl=15,
    show_spinner=False,
)
def api_health():

    response = api_request(
        "GET",
        "/health",
        retries=3,
        timeout=(
            API_CONNECT_TIMEOUT,
            20,
        ),
    )

    if response is None:
        return {}

    try:

        data = response.json()

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception:
        pass

    return {}


# ============================================================
# LOCAL FALLBACK EVENTS
# ============================================================

def get_local_events(
    limit=100
):

    if ml_df.empty:
        return pd.DataFrame()

    try:

        df = ml_df.copy()

        if "event_date" in df.columns:

            df["event_date"] = pd.to_datetime(
                df["event_date"],
                errors="coerce",
            )

        if "event_id" in df.columns:

            df["event_id"] = pd.to_numeric(
                df["event_id"],
                errors="coerce",
            )

        if "event_date" in df.columns:

            df = df.sort_values(
                "event_date",
                ascending=False,
            )

        return df.head(
            int(limit)
        ).copy()

    except Exception:

        return pd.DataFrame()


# ============================================================
# EVENTS FROM FASTAPI
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_events_from_api(
    limit=100
):

    response = api_request(
        "GET",
        "/events",
        params={
            "limit": int(limit),
            "sort_by": "latest",
        },
        retries=3,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    if response is not None:

        try:

            data = response.json()

            if isinstance(
                data,
                dict,
            ):

                if "events" in data:

                    data = data[
                        "events"
                    ]

                elif "data" in data:

                    data = data[
                        "data"
                    ]

            df = pd.DataFrame(
                data
            )

            if not df.empty:
                return df

        except Exception:
            pass

    # --------------------------------------------------------
    # IMPORTANT FALLBACK
    #
    # If Render temporarily returns 502/503/504,
    # the dashboard can still display local event data.
    # --------------------------------------------------------

    return get_local_events(
        limit
    )


# ============================================================
# ML PREDICTION
# ============================================================

def get_prediction(
    event_id
):

    response = api_request(
        "POST",
        "/predict-event",
        json={
            "event_id": int(
                event_id
            )
        },
        retries=3,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    if response is None:
        return None

    if not response.ok:
        return None

    try:
        return response.json()

    except Exception:
        return None


# ============================================================
# EVENT ANALYSIS
# ============================================================

def get_event_analysis(
    event_id
):

    response = api_request(
        "GET",
        f"/event-analysis/{int(event_id)}",
        retries=3,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    if response is None:
        return None

    if not response.ok:
        return None

    try:
        return response.json()

    except Exception:
        return None


# ============================================================
# SENTINEL EVIDENCE
# ============================================================

def get_sentinel_evidence(
    event_id
):

    response = api_request(
        "GET",
        f"/sentinel/{int(event_id)}/evidence",
        retries=2,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    if (
        response is not None
        and response.ok
    ):

        try:

            data = response.json()

            if isinstance(
                data,
                dict,
            ):

                evidence = data.get(
                    "evidence"
                )

                if isinstance(
                    evidence,
                    dict,
                ):

                    sentinel_info = data.get(
                        "sentinel",
                        {},
                    )

                    evidence[
                        "_sentinel_available"
                    ] = sentinel_info.get(
                        "available",
                        False,
                    )

                    evidence[
                        "_sentinel_source"
                    ] = sentinel_info.get(
                        "source",
                        "unknown",
                    )

                    return evidence

                return data

        except Exception:
            pass

    analysis = get_event_analysis(
        event_id
    )

    if analysis:

        sentinel = analysis.get(
            "sentinel"
        )

        if isinstance(
            sentinel,
            dict,
        ):

            nested = sentinel.get(
                "evidence"
            )

            if isinstance(
                nested,
                dict,
            ):

                return nested

            return sentinel

    return None


# ============================================================
# SENTINEL VISUALIZATIONS
# ============================================================

def get_sentinel_visualizations(
    event_id
):

    result = {}

    visualization_types = [
        "ndvi",
        "nbr",
        "scl",
        "false_color",
    ]

    for visualization_type in (
        visualization_types
    ):

        response = api_request(
            "GET",
            (
                f"/sentinel/{int(event_id)}"
                f"/visualizations/"
                f"{visualization_type}"
            ),
            retries=2,
            timeout=(
                API_CONNECT_TIMEOUT,
                API_READ_TIMEOUT,
            ),
        )

        if (
            response is not None
            and response.ok
            and response.content
        ):

            result[
                visualization_type
            ] = response.content

    visualization_dir = (
        BASE_DIR
        / "data"
        / "sentinel"
        / f"event_{int(event_id)}"
        / "visualizations"
    )

    local_files = {

        "ndvi":
            visualization_dir
            / f"event_{int(event_id)}_NDVI.png",

        "nbr":
            visualization_dir
            / f"event_{int(event_id)}_NBR.png",

        "scl":
            visualization_dir
            / f"event_{int(event_id)}_SCL.png",

        "false_color":
            visualization_dir
            / f"event_{int(event_id)}_false_color.png",
    }

    for key, path in (
        local_files.items()
    ):

        if key in result:
            continue

        if path.exists():

            try:

                result[
                    key
                ] = path.read_bytes()

            except Exception:
                pass

    return result


# ============================================================
# LOAD DATA
# ============================================================

ml_df = load_ml_data()

osm_df = load_osm_data()

health = api_health()

api_connected = (
    health.get(
        "status"
    )
    == "healthy"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔥 VEYRONIX"
    )

    st.caption(
        "Industrial Fire Detection & "
        "Source Attribution"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Fire Map",
            "Event Investigation",
            "Analytics",
            "Satellite Evidence",
            "About",
        ],
    )

    st.divider()

    st.markdown(
        "### System Status"
    )

    if api_connected:

        st.success(
            "FastAPI connected"
        )

    else:

        st.error(
            "FastAPI unavailable"
        )

        if st.session_state.get(
            "api_last_error"
        ):

            st.caption(
                (
                    "API: "
                    f"{st.session_state['api_last_status']}"
                )
            )

    if health.get(
        "model_available",
        False,
    ):

        st.success(
            "LightGBM model ready"
        )

    else:

        st.warning(
            "LightGBM unavailable"
        )

    if health.get(
        "dataset_available",
        False,
    ):

        st.success(
            "Backend dataset ready"
        )

    elif not ml_df.empty:

        st.info(
            f"{len(ml_df):,} "
            "thermal events loaded locally"
        )

    else:

        st.error(
            "Dataset unavailable"
        )

    if health.get(
        "api_key_configured",
        False,
    ):

        st.success(
            "API authentication enabled"
        )

    else:

        st.warning(
            "Local demo authentication"
        )

    st.divider()

    st.caption(
        f"API: {API_URL}"
    )

    st.caption(
        "SIH26162"
    )

    st.caption(
        "VEYRONIX AI"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🔥 VEYRONIX AI"
)

st.caption(
    "Industrial Fire Detection & Source Attribution "
    "using NASA FIRMS, OSM and Sentinel-2 evidence."
)

st.divider()


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.header(
        "Command Center"
    )

    if ml_df.empty:

        st.error(
            "ML dataset could not be loaded."
        )

        st.stop()

    total_events = len(
        ml_df
    )

    anomaly_count = 0

    persistent_count = 0

    peak_frp = 0

    if "is_anomaly" in ml_df.columns:

        anomaly_count = (
            ml_df[
                "is_anomaly"
            ]
            .apply(
                safe_bool
            )
            .sum()
        )

    if "persistent" in ml_df.columns:

        persistent_count = (
            ml_df[
                "persistent"
            ]
            .apply(
                safe_bool
            )
            .sum()
        )

    if "peak_frp" in ml_df.columns:

        peak_frp = pd.to_numeric(
            ml_df[
                "peak_frp"
            ],
            errors="coerce",
        ).max()

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Thermal Events",
            f"{total_events:,}",
        )

    with c2:

        st.metric(
            "Anomalies",
            f"{int(anomaly_count):,}",
        )

    with c3:

        st.metric(
            "Persistent Events",
            f"{int(persistent_count):,}",
        )

    with c4:

        st.metric(
            "Peak FRP",
            f"{peak_frp:.2f} MW",
        )

    st.subheader(
        "Recent Thermal Activity"
    )

    recent = get_events_from_api(
        100
    )

    if recent.empty:

        st.warning(
            "No recent events are currently available."
        )

        if st.session_state.get(
            "api_last_error"
        ):

            st.caption(
                (
                    "FastAPI status: "
                    f"{st.session_state['api_last_status']} — "
                    f"{st.session_state['api_last_error']}"
                )
            )

    else:

        # Detect whether we are showing local fallback
        # instead of the API response.

        if (
            not api_connected
            or st.session_state.get(
                "api_last_status"
            ) in {
                "502",
                "503",
                "504",
                "Connection failed",
            }
        ):

            st.info(
                "FastAPI is temporarily unavailable. "
                "Showing the local thermal-event dataset."
            )

        display = recent.copy()

        if "event_date" in display.columns:

            display[
                "event_date"
            ] = pd.to_datetime(
                display[
                    "event_date"
                ],
                errors="coerce",
            )

        if "event_id" in display.columns:

            display[
                "event_id"
            ] = pd.to_numeric(
                display[
                    "event_id"
                ],
                errors="coerce",
            )

        display[
            "priority"
        ] = display.apply(
            priority_label,
            axis=1,
        )

        columns = [
            "event_id",
            "event_date",
            "latitude",
            "longitude",
            "observation_count",
            "peak_frp",
            "priority",
        ]

        columns = [
            x
            for x in columns
            if x in display.columns
        ]

        display = display[
            columns
        ].copy()

        if "event_date" in display.columns:

            display[
                "event_date"
            ] = display[
                "event_date"
            ].dt.strftime(
                "%Y-%m-%d"
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Quick Investigation"
    )

    selected_event = st.number_input(
        "Event ID",
        min_value=1,
        value=int(
            st.session_state.get(
                "selected_event",
                84118,
            )
        ),
        step=1,
    )

    if st.button(
        "🔎 Investigate Event",
        type="primary",
    ):

        st.session_state[
            "selected_event"
        ] = int(
            selected_event
        )

        st.info(
            "Open Event Investigation "
            "from the sidebar."
        )


# ============================================================
# FIRE MAP
# ============================================================

elif page == "Fire Map":

    st.header(
        "🔥 Fire Map"
    )

    recent = get_events_from_api(
        300
    )

    if recent.empty:

        st.warning(
            "No event data is currently available."
        )

        if st.session_state.get(
            "api_last_error"
        ):

            st.caption(
                (
                    "FastAPI: "
                    f"{st.session_state['api_last_status']} — "
                    f"{st.session_state['api_last_error']}"
                )
            )

        st.stop()

    show_anomaly = st.checkbox(
        "Show anomalies only"
    )

    show_persistent = st.checkbox(
        "Show persistent events only"
    )

    min_frp = st.number_input(
        "Minimum Peak FRP",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    filtered = recent.copy()

    if (
        show_anomaly
        and "is_anomaly"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "is_anomaly"
            ].apply(
                safe_bool
            )
        ]

    if (
        show_persistent
        and "persistent"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "persistent"
            ].apply(
                safe_bool
            )
        ]

    if "peak_frp" in filtered.columns:

        filtered[
            "peak_frp"
        ] = pd.to_numeric(
            filtered[
                "peak_frp"
            ],
            errors="coerce",
        )

        filtered = filtered[
            filtered[
                "peak_frp"
            ] >= min_frp
        ]

    if filtered.empty:

        st.info(
            "No events match the filters."
        )

    else:

        center_lat = pd.to_numeric(
            filtered[
                "latitude"
            ],
            errors="coerce",
        ).mean()

        center_lon = pd.to_numeric(
            filtered[
                "longitude"
            ],
            errors="coerce",
        ).mean()

        fmap = folium.Map(
            location=[
                center_lat,
                center_lon,
            ],
            zoom_start=5,
            tiles=None,
        )

        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/"
                "ArcGIS/rest/services/"
                "World_Imagery/"
                "MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Esri World Imagery",
            name="Satellite",
        ).add_to(
            fmap
        )

        folium.TileLayer(
            "OpenStreetMap",
            name="Road Map",
        ).add_to(
            fmap
        )

        for _, row in (
            filtered.iterrows()
        ):

            lat = row.get(
                "latitude"
            )

            lon = row.get(
                "longitude"
            )

            if pd.isna(lat) or pd.isna(lon):
                continue

            anomaly = safe_bool(
                row.get(
                    "is_anomaly",
                    False,
                )
            )

            persistent = safe_bool(
                row.get(
                    "persistent",
                    False,
                )
            )

            try:

                frp = float(
                    row.get(
                        "peak_frp",
                        0,
                    )
                    or 0
                )

            except Exception:

                frp = 0

            if anomaly:

                marker_color = "red"

            elif persistent:

                marker_color = "orange"

            else:

                marker_color = "blue"

            event_id = int(
                row[
                    "event_id"
                ]
            )

            popup = f"""
            <b>VEYRONIX AI</b><br>
            Event ID: {event_id}<br>
            Date: {row.get("event_date", "")}<br>
            Peak FRP: {frp:.2f} MW<br>
            Anomaly: {anomaly}<br>
            Persistent: {persistent}
            """

            folium.CircleMarker(
                location=[
                    lat,
                    lon,
                ],
                radius=6,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                popup=popup,
                tooltip=(
                    f"Event {event_id}"
                ),
            ).add_to(
                fmap
            )

        folium.LayerControl().add_to(
            fmap
        )

        st_folium(
            fmap,
            width=None,
            height=620,
        )

        st.caption(
            f"Showing {len(filtered):,} events. "
            "Red = anomaly • Orange = persistent • "
            "Blue = normal"
        )


# ============================================================
# EVENT INVESTIGATION
# ============================================================

elif page == "Event Investigation":

    st.header(
        "🔎 Event Investigation"
    )

    default_event = (
        st.session_state.get(
            "selected_event",
            84118,
        )
    )

    event_id = st.number_input(
        "Event ID",
        min_value=1,
        value=int(
            default_event
        ),
        step=1,
    )

    if st.button(
        "Run Investigation",
        type="primary",
    ):

        with st.spinner(
            "Running VEYRONIX investigation..."
        ):

            prediction = get_prediction(
                event_id
            )

            analysis = get_event_analysis(
                event_id
            )

        if prediction is None:

            st.error(
                "Prediction could not be retrieved "
                "from FastAPI."
            )

            if st.session_state.get(
                "api_last_error"
            ):

                st.warning(
                    (
                        f"FastAPI returned "
                        f"{st.session_state['api_last_status']}. "
                        f"{st.session_state['api_last_error']}"
                    )
                )

            st.info(
                "If Render recently went to sleep, "
                "wait a few seconds and run the investigation again."
            )

            st.stop()

        # ====================================================
        # EVENT DATA
        # ====================================================

        event_row = pd.Series(
            dtype=object
        )

        if (
            not ml_df.empty
            and "event_id"
            in ml_df.columns
        ):

            matches = ml_df[
                ml_df[
                    "event_id"
                ] == event_id
            ]

            if not matches.empty:

                event_row = matches.iloc[
                    0
                ]

        # ====================================================
        # ML PREDICTION
        # ====================================================

        prediction_block = prediction.get(
            "prediction",
            {},
        )

        source = normalize_source(
            prediction_block.get(
                "predicted_source",
                "Uncertain",
            )
        )

        confidence = float(
            prediction_block.get(
                "confidence",
                0,
            )
        )

        location = prediction.get(
            "location",
            {},
        )

        latitude = location.get(
            "latitude"
        )

        longitude = location.get(
            "longitude"
        )

        # ====================================================
        # OSM CANDIDATE EVIDENCE
        # ====================================================

        osm_sources = []

        if (
            not osm_df.empty
            and "event_id"
            in osm_df.columns
        ):

            candidates = osm_df[
                osm_df[
                    "event_id"
                ] == event_id
            ].copy()

            if (
                not candidates.empty
                and "source_candidate"
                in candidates.columns
            ):

                osm_sources = (
                    candidates[
                        "source_candidate"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

        # ====================================================
        # RECONCILE
        # ====================================================

        reconciliation = (
            reconcile_evidence(
                source,
                confidence,
                osm_sources,
            )
        )

        attribution_status = (
            reconciliation[
                "status"
            ]
        )

        # ====================================================
        # SOURCE ATTRIBUTION
        # ====================================================

        st.subheader(
            f"{source_icon(source)} "
            "Source Attribution"
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            st.metric(
                "Primary Source",
                source,
            )

        with c2:

            st.metric(
                "ML Confidence",
                f"{confidence * 100:.2f}%",
            )

        with c3:

            st.metric(
                "Attribution Status",
                attribution_status,
            )

        st.caption(
            "LightGBM provides the primary source "
            "prediction. OSM and Sentinel-2 are "
            "supporting contextual evidence."
        )

        # ====================================================
        # EVIDENCE RECONCILIATION
        # ====================================================

        st.subheader(
            "Evidence Reconciliation"
        )

        if (
            reconciliation["level"]
            == "success"
        ):

            st.success(
                "✅ **"
                + reconciliation[
                    "status"
                ]
                + "**\n\n"
                + reconciliation[
                    "message"
                ]
            )

        elif (
            reconciliation["level"]
            == "warning"
        ):

            st.warning(
                "⚠️ **"
                + reconciliation[
                    "status"
                ]
                + "**\n\n"
                + reconciliation[
                    "message"
                ]
            )

        else:

            st.info(
                "ℹ️ **"
                + reconciliation[
                    "status"
                ]
                + "**\n\n"
                + reconciliation[
                    "message"
                ]
            )

        # ====================================================
        # SOURCE PROBABILITIES
        # ====================================================

        probabilities = (
            prediction_block.get(
                "probabilities",
                {},
            )
        )

        if probabilities:

            st.subheader(
                "Source Probability Distribution"
            )

            probability_df = pd.DataFrame(
                {
                    "Source": [
                        normalize_source(
                            key
                        )
                        for key
                        in probabilities.keys()
                    ],

                    "Probability": [
                        float(value)
                        for value
                        in probabilities.values()
                    ],
                }
            )

            probability_df = (
                probability_df.sort_values(
                    "Probability",
                    ascending=False,
                )
            )

            st.bar_chart(
                probability_df.set_index(
                    "Source"
                )
            )

            display_probability = (
                probability_df.copy()
            )

            display_probability[
                "Probability"
            ] = (
                display_probability[
                    "Probability"
                ]
                * 100
            ).round(2)

            st.dataframe(
                display_probability,
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # THERMAL CHARACTERISTICS
        # ====================================================

        st.subheader(
            "🔥 Thermal Characteristics"
        )

        if not event_row.empty:

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            with c1:

                st.metric(
                    "Observations",
                    int(
                        event_row.get(
                            "observation_count",
                            0,
                        )
                    ),
                )

            with c2:

                st.metric(
                    "Mean FRP",
                    (
                        f"{float(event_row.get('mean_frp', 0)):.2f} MW"
                    ),
                )

            with c3:

                st.metric(
                    "Peak FRP",
                    (
                        f"{float(event_row.get('peak_frp', 0)):.2f} MW"
                    ),
                )

            with c4:

                st.metric(
                    "Total FRP",
                    (
                        f"{float(event_row.get('total_frp', 0)):.2f} MW"
                    ),
                )

        # ====================================================
        # LOCATION
        # ====================================================

        st.subheader(
            "📍 Detected Hotspot Location"
        )

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            st.metric(
                "Latitude",
                (
                    f"{float(latitude):.6f}"
                    if latitude is not None
                    else "N/A"
                ),
            )

        with c2:

            st.metric(
                "Longitude",
                (
                    f"{float(longitude):.6f}"
                    if longitude is not None
                    else "N/A"
                ),
            )

        if (
            latitude is not None
            and longitude is not None
        ):

            location_df = pd.DataFrame(
                {
                    "latitude": [
                        latitude
                    ],
                    "longitude": [
                        longitude
                    ],
                }
            )

            st.map(
                location_df,
                zoom=10,
            )

            google_url = (
                "https://www.google.com/maps/search/"
                f"?api=1&query={latitude},{longitude}"
            )

            osm_url = (
                "https://www.openstreetmap.org/"
                f"?mlat={latitude}"
                f"&mlon={longitude}"
                "&zoom=15"
            )

            st.markdown(
                f"[🌍 Open in Google Maps]({google_url})"
            )

            st.markdown(
                f"[🗺️ Open in OpenStreetMap]({osm_url})"
            )

            st.caption(
                "Location represents a satellite-detected "
                "thermal hotspot, not necessarily an exact "
                "building-level fire location."
            )

        # ====================================================
        # OSM EVIDENCE
        # ====================================================

        st.subheader(
            "🗺️ OSM Supporting Evidence"
        )

        osm_found = False

        if (
            not osm_df.empty
            and "event_id"
            in osm_df.columns
        ):

            candidates = osm_df[
                osm_df[
                    "event_id"
                ] == event_id
            ].copy()

            if not candidates.empty:

                osm_found = True

                st.success(
                    "OSM contextual evidence found "
                    "for this event."
                )

                if (
                    "source_candidate"
                    in candidates.columns
                ):

                    osm_sources = (
                        candidates[
                            "source_candidate"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    for candidate in (
                        osm_sources
                    ):

                        st.info(
                            "OSM candidate context: "
                            f"**{normalize_source(candidate)}**"
                        )

                st.dataframe(
                    candidates,
                    use_container_width=True,
                    hide_index=True,
                )

        if (
            not osm_found
            and latitude is not None
            and longitude is not None
        ):

            spatial_osm = (
                get_osm_spatial_evidence(
                    latitude,
                    longitude,
                    radius_km=2.0,
                )
            )

            if not spatial_osm.empty:

                osm_found = True

                st.success(
                    "Nearby OSM land-use context "
                    "found within approximately 2 km."
                )

                st.dataframe(
                    spatial_osm,
                    use_container_width=True,
                    hide_index=True,
                )

                spatial_sources = (
                    spatial_osm[
                        "source_context"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                osm_sources = (
                    spatial_sources
                )

        if not osm_found:

            st.info(
                "No OSM contextual evidence "
                "was found for this event."
            )

        st.caption(
            "OSM provides contextual land-use evidence "
            "and is not treated as independent ground truth."
        )

        # ====================================================
        # HISTORICAL BEHAVIOUR
        # ====================================================

        st.subheader(
            "📈 Historical Behaviour"
        )

        if not event_row.empty:

            c1, c2, c3 = (
                st.columns(3)
            )

            with c1:

                st.metric(
                    "Historical Detections",
                    int(
                        event_row.get(
                            "historical_detection_count",
                            0,
                        )
                    ),
                )

            with c2:

                st.metric(
                    "Historical Mean FRP",
                    (
                        f"{float(event_row.get('historical_mean_frp', 0)):.2f}"
                    ),
                )

            with c3:

                previously_detected = (
                    safe_bool(
                        event_row.get(
                            "previously_detected",
                            False,
                        )
                    )
                )

                st.metric(
                    "Previously Detected",
                    (
                        "Yes"
                        if previously_detected
                        else "No"
                    ),
                )

        # ====================================================
        # ANOMALY
        # ====================================================

        st.subheader(
            "🚨 Anomaly Detection"
        )

        anomaly = (
            prediction.get(
                "anomaly",
                {},
            )
        )

        if not anomaly and analysis:

            anomaly = (
                analysis.get(
                    "anomaly",
                    {},
                )
            )

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            is_anomaly = safe_bool(
                anomaly.get(
                    "is_anomaly",
                    False,
                )
            )

            st.metric(
                "Anomaly Status",
                (
                    "ANOMALY"
                    if is_anomaly
                    else "NORMAL"
                ),
            )

        with c2:

            try:

                anomaly_score = float(
                    anomaly.get(
                        "anomaly_score",
                        0,
                    )
                )

            except Exception:

                anomaly_score = 0

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.4f}",
            )

        # ====================================================
        # SENTINEL-2
        # ====================================================

        st.subheader(
            "🛰️ Sentinel-2 Evidence"
        )

        sentinel = (
            get_sentinel_evidence(
                event_id
            )
        )

        images = (
            get_sentinel_visualizations(
                event_id
            )
        )

        if sentinel:

            st.success(
                "Sentinel-2 evidence available."
            )

            ndvi = sentinel.get(
                "ndvi",
                {},
            )

            nbr = sentinel.get(
                "nbr",
                {},
            )

            scl = sentinel.get(
                "scl",
                {},
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            with c1:

                st.metric(
                    "NDVI Mean",
                    format_number(
                        ndvi.get(
                            "mean"
                        )
                        if isinstance(
                            ndvi,
                            dict,
                        )
                        else None,
                        4,
                    ),
                )

            with c2:

                st.metric(
                    "NBR Mean",
                    format_number(
                        nbr.get(
                            "mean"
                        )
                        if isinstance(
                            nbr,
                            dict,
                        )
                        else None,
                        4,
                    ),
                )

            with c3:

                st.metric(
                    "SCL Center",
                    str(
                        scl.get(
                            "center_class",
                            "N/A",
                        )
                        if isinstance(
                            scl,
                            dict,
                        )
                        else "N/A"
                    ),
                )

            with c4:

                st.metric(
                    "Evidence Source",
                    str(
                        sentinel.get(
                            "_sentinel_source",
                            "Available",
                        )
                    ),
                )

            if int(event_id) == 84118:

                st.info(
                    "The available Sentinel-2 scene "
                    "for Event 84118 is approximately "
                    "2.23 days after the FIRMS detection. "
                    "It should therefore be interpreted "
                    "as contextual evidence rather than "
                    "simultaneous fire confirmation."
                )

            else:

                st.info(
                    "Sentinel-2 provides contextual "
                    "evidence and may not be temporally "
                    "simultaneous with the FIRMS observation."
                )

            with st.expander(
                "View Sentinel Metrics"
            ):

                display_sentinel = {
                    key: value
                    for key, value in sentinel.items()
                    if not str(
                        key
                    ).startswith("_")
                    and key != "files"
                }

                st.json(
                    display_sentinel
                )

        else:

            st.warning(
                "No Sentinel-2 evidence is "
                "currently available for this event."
            )

        # ====================================================
        # SENTINEL IMAGES
        # ====================================================

        if images:

            st.markdown(
                "### 🛰️ Sentinel-2 Visual Evidence"
            )

            tabs = st.tabs(
                [
                    "False Color",
                    "NDVI",
                    "NBR",
                    "SCL",
                ]
            )

            image_mapping = [
                (
                    "false_color",
                    "False Color",
                ),
                (
                    "ndvi",
                    "NDVI",
                ),
                (
                    "nbr",
                    "NBR",
                ),
                (
                    "scl",
                    "SCL",
                ),
            ]

            for tab, (
                key,
                title,
            ) in zip(
                tabs,
                image_mapping,
            ):

                with tab:

                    if key in images:

                        st.image(
                            images[
                                key
                            ],
                            caption=(
                                f"Sentinel-2 "
                                f"{title} — "
                                f"Event "
                                f"{event_id}"
                            ),
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            f"{title} visualization "
                            "is not available."
                        )

        else:

            st.info(
                "No Sentinel-2 visualization "
                "images are currently available."
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.header(
        "📊 Analytics"
    )

    if ml_df.empty:

        st.error(
            "ML dataset unavailable."
        )

        st.stop()

    # --------------------------------------------------------
    # OSM
    # --------------------------------------------------------

    st.subheader(
        "OSM Candidate Source Distribution"
    )

    if (
        not osm_df.empty
        and "source_candidate"
        in osm_df.columns
    ):

        source_counts = (
            osm_df[
                "source_candidate"
            ]
            .value_counts()
            .rename_axis(
                "Source"
            )
            .reset_index(
                name="Events"
            )
        )

        source_counts[
            "Source"
        ] = source_counts[
            "Source"
        ].apply(
            normalize_source
        )

        st.bar_chart(
            source_counts.set_index(
                "Source"
            )
        )

        st.dataframe(
            source_counts,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "OSM candidate dataset unavailable."
        )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    st.subheader(
        "Anomaly Distribution"
    )

    if "is_anomaly" in ml_df.columns:

        normal_count = (
            ~ml_df[
                "is_anomaly"
            ]
            .apply(
                safe_bool
            )
        ).sum()

        anomaly_count = (
            ml_df[
                "is_anomaly"
            ]
            .apply(
                safe_bool
            )
        ).sum()

        anomaly_counts = pd.DataFrame(
            {
                "Status": [
                    "Normal",
                    "Anomaly",
                ],

                "Events": [
                    normal_count,
                    anomaly_count,
                ],
            }
        )

        st.bar_chart(
            anomaly_counts.set_index(
                "Status"
            )
        )

    # --------------------------------------------------------
    # FRP
    # --------------------------------------------------------

    st.subheader(
        "Peak FRP Distribution"
    )

    if "peak_frp" in ml_df.columns:

        frp = pd.to_numeric(
            ml_df[
                "peak_frp"
            ],
            errors="coerce",
        ).dropna()

        histogram = (
            pd.cut(
                frp,
                bins=[
                    0,
                    1,
                    5,
                    10,
                    20,
                    50,
                    100,
                    np.inf,
                ],
                labels=[
                    "0–1",
                    "1–5",
                    "5–10",
                    "10–20",
                    "20–50",
                    "50–100",
                    "100+",
                ],
            )
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            histogram
        )


# ============================================================
# SATELLITE EVIDENCE PAGE
# ============================================================

elif page == "Satellite Evidence":

    st.header(
        "🛰️ Sentinel-2 Evidence"
    )

    st.write(
        "Sentinel-2 provides higher-resolution "
        "context around NASA FIRMS thermal hotspots."
    )

    st.info(
        "Sentinel-2 is supporting contextual evidence "
        "and should not automatically be treated as "
        "ground truth."
    )

    event_id = st.number_input(
        "Event ID",
        min_value=1,
        value=84118,
        step=1,
    )

    if st.button(
        "Load Sentinel Evidence",
        type="primary",
    ):

        with st.spinner(
            "Loading Sentinel-2 evidence..."
        ):

            sentinel = (
                get_sentinel_evidence(
                    event_id
                )
            )

            images = (
                get_sentinel_visualizations(
                    event_id
                )
            )

        if (
            not sentinel
            and not images
        ):

            st.warning(
                "No Sentinel-2 evidence "
                "is available for this event."
            )

        else:

            if sentinel:

                ndvi = sentinel.get(
                    "ndvi",
                    {},
                )

                nbr = sentinel.get(
                    "nbr",
                    {},
                )

                scl = sentinel.get(
                    "scl",
                    {},
                )

                c1, c2, c3 = (
                    st.columns(3)
                )

                with c1:

                    st.metric(
                        "NDVI Mean",
                        format_number(
                            ndvi.get(
                                "mean"
                            )
                            if isinstance(
                                ndvi,
                                dict,
                            )
                            else None,
                            4,
                        ),
                    )

                with c2:

                    st.metric(
                        "NBR Mean",
                        format_number(
                            nbr.get(
                                "mean"
                            )
                            if isinstance(
                                nbr,
                                dict,
                            )
                            else None,
                            4,
                        ),
                    )

                with c3:

                    st.metric(
                        "SCL Center",
                        str(
                            scl.get(
                                "center_class",
                                "N/A",
                            )
                            if isinstance(
                                scl,
                                dict,
                            )
                            else "N/A"
                        ),
                    )

            if images:

                st.subheader(
                    "Satellite Visualizations"
                )

                tabs = st.tabs(
                    [
                        "False Color",
                        "NDVI",
                        "NBR",
                        "SCL",
                    ]
                )

                image_mapping = [
                    (
                        "false_color",
                        "False Color",
                    ),
                    (
                        "ndvi",
                        "NDVI",
                    ),
                    (
                        "nbr",
                        "NBR",
                    ),
                    (
                        "scl",
                        "SCL",
                    ),
                ]

                for tab, (
                    key,
                    title,
                ) in zip(
                    tabs,
                    image_mapping,
                ):

                    with tab:

                        if key in images:

                            st.image(
                                images[
                                    key
                                ],
                                caption=(
                                    f"Sentinel-2 "
                                    f"{title}"
                                ),
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                f"{title} image "
                                "unavailable."
                            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header(
        "About VEYRONIX AI"
    )

    st.write(
        "VEYRONIX AI is a decision-support platform "
        "for detecting and attributing persistent "
        "thermal sources using NASA FIRMS, machine "
        "learning, OSM context and Sentinel-2 evidence."
    )

    st.subheader(
        "System Pipeline"
    )

    st.markdown(
        """
        **NASA FIRMS / VIIRS**

        ↓

        **Thermal Event Construction**

        ↓

        **Feature Engineering**

        ↓

        **Anomaly Detection**

        ↓

        **LightGBM Source Attribution**

        ↓

        **OSM Supporting Context**

        ↓

        **Sentinel-2 Evidence**

        ↓

        **Evidence Reconciliation**

        ↓

        **FastAPI**

        ↓

        **VEYRONIX Command Center**
        """
    )

    st.subheader(
        "Source Classes"
    )

    classes = pd.DataFrame(
        {
            "Class": [
                "Industrial",
                "Agriculture / Biomass",
                "Forest / Natural",
                "Waste / Other",
                "Uncertain",
            ],

            "Meaning": [

                "Industrial facilities "
                "and related thermal activity",

                "Crop residue, agricultural "
                "burning and biomass activity",

                "Forest or naturally occurring "
                "thermal activity",

                "Waste, landfill and other "
                "contextual sources",

                "Evidence is insufficient "
                "for confident attribution",
            ],
        }
    )

    st.dataframe(
        classes,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "A NASA FIRMS thermal hotspot is not automatically "
        "an exact building-level fire location. VEYRONIX "
        "combines ML prediction with spatial, historical, "
        "OSM and Sentinel-2 evidence."
    )

    st.subheader(
        "Evidence Reconciliation"
    )

    st.markdown(
        """
        **MODEL SUPPORTED**

        ML prediction and OSM contextual evidence are
        consistent.

        **NEEDS REVIEW**

        ML prediction and contextual evidence disagree.
        The system does not automatically override the
        ML prediction.

        **LOW CONFIDENCE**

        The model confidence is low and additional
        evidence should be reviewed.

        **MODEL PREDICTION**

        A model prediction is available but there is
        insufficient contextual evidence for reconciliation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔥 VEYRONIX AI • SIH26162 • "
    "NASA FIRMS + OSM + Sentinel-2 + LightGBM"
)