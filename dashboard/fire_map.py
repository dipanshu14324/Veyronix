import streamlit as st
import requests
import folium

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH26162 Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

PREDICT_TIMEOUT = 30
SENTINEL_TIMEOUT = 120


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .evidence-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.05);
        margin-bottom: 1rem;
    }

    .status-good {
        color: #15803d;
        font-weight: 600;
    }

    .status-warning {
        color: #b45309;
        font-weight: 600;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "event_result" not in st.session_state:
    st.session_state.event_result = None

if "sentinel_result" not in st.session_state:
    st.session_state.sentinel_result = None

if "last_event_id" not in st.session_state:
    st.session_state.last_event_id = None


# ============================================================
# API HELPERS
# ============================================================

def predict_event(event_id: int):
    """
    Ask FastAPI to analyze the thermal event.
    """

    response = requests.post(
        f"{API_URL}/predict-event",
        json={
            "event_id": int(event_id)
        },
        timeout=PREDICT_TIMEOUT
    )

    if response.status_code != 200:

        try:
            error_detail = response.json()

        except Exception:
            error_detail = response.text

        raise RuntimeError(
            f"Prediction API returned "
            f"HTTP {response.status_code}: "
            f"{error_detail}"
        )

    return response.json()


def get_sentinel_evidence(event_id: int):
    """
    Get Sentinel-2 evidence from FastAPI.

    Streamlit does NOT directly read Sentinel JP2 files.
    FastAPI handles:
        Sentinel Manager
        local cache
        B04/B08/B12/SCL
        NDVI
        NBR
        SCL analysis
    """

    response = requests.get(
        f"{API_URL}/sentinel/{int(event_id)}/evidence",
        timeout=SENTINEL_TIMEOUT
    )

    if response.status_code == 404:
        return None

    if response.status_code != 200:

        try:
            error_detail = response.json()

        except Exception:
            error_detail = response.text

        raise RuntimeError(
            f"Sentinel API returned "
            f"HTTP {response.status_code}: "
            f"{error_detail}"
        )

    return response.json()


# ============================================================
# FIRMS MAP
# ============================================================

def create_firms_map(
    latitude,
    longitude,
    event_id,
    source,
    confidence,
    event
):
    """
    Create the basic FIRMS hotspot map.
    """

    fire_map = folium.Map(
        location=[
            latitude,
            longitude
        ],
        zoom_start=15,
        control_scale=True,
        tiles="OpenStreetMap"
    )

    # --------------------------------------------------------
    # FIRMS HOTSPOT MARKER
    # --------------------------------------------------------

    popup_html = f"""
    <div style="width:300px">

        <h4>🔥 FIRMS Thermal Hotspot</h4>

        <b>Event ID:</b> {event_id}<br>

        <b>Latitude:</b> {latitude:.5f}<br>

        <b>Longitude:</b> {longitude:.5f}<br><br>

        <b>Predicted Source:</b><br>
        {source}<br>

        <b>Model Score:</b><br>
        {confidence * 100:.2f}%<br><br>

        <b>Mean FRP:</b>
        {event.get("mean_frp", "N/A")}<br>

        <b>Peak FRP:</b>
        {event.get("peak_frp", "N/A")}<br>

        <b>Observations:</b>
        {event.get("observation_count", "N/A")}

    </div>
    """

    folium.Marker(
        location=[
            latitude,
            longitude
        ],
        tooltip=f"🔥 FIRMS Event {event_id}",
        popup=folium.Popup(
            popup_html,
            max_width=350
        ),
        icon=folium.Icon(
            color="red",
            icon="fire",
            prefix="fa"
        )
    ).add_to(fire_map)

    # --------------------------------------------------------
    # CIRCLE SHOWING APPROXIMATE LOCATION CONTEXT
    # --------------------------------------------------------

    folium.Circle(
        location=[
            latitude,
            longitude
        ],
        radius=375,
        tooltip="Approximate VIIRS pixel-scale context",
        color="red",
        fill=False,
        weight=2
    ).add_to(fire_map)

    # --------------------------------------------------------
    # MAP LAYER
    # --------------------------------------------------------

    folium.LayerControl().add_to(fire_map)

    return fire_map


# ============================================================
# DISPLAY SENTINEL EVIDENCE
# ============================================================

def display_sentinel_evidence(
    sentinel_data,
    event_id
):
    """
    Display Sentinel-2 evidence returned by FastAPI.
    """

    st.subheader(
        "🛰️ Sentinel-2 Satellite Evidence"
    )

    # --------------------------------------------------------
    # BASIC RESPONSE VALIDATION
    # --------------------------------------------------------

    if not sentinel_data:

        st.warning(
            f"⚠️ Sentinel-2 evidence is not available "
            f"for Event {event_id}."
        )

        return

    if sentinel_data.get("status") != "success":

        st.warning(
            "⚠️ Sentinel-2 evidence could not be loaded."
        )

        return

    evidence = sentinel_data.get(
        "evidence",
        {}
    )

    sentinel_status = sentinel_data.get(
        "sentinel",
        {}
    )

    location = sentinel_data.get(
        "location",
        {}
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    source = sentinel_status.get(
        "source",
        "unknown"
    )

    available = sentinel_status.get(
        "available",
        False
    )

    if available:

        if source == "local_cache":

            st.success(
                "✓ Sentinel-2 evidence loaded "
                "from local cache."
            )

        else:

            st.success(
                "✓ Sentinel-2 evidence is available."
            )

    else:

        st.warning(
            "⚠️ Sentinel-2 data is not currently available."
        )

        return

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="evidence-box">

        <b>Sentinel Evidence Location</b><br>

        Latitude:
        {float(location.get("latitude", 0)):.5f}<br>

        Longitude:
        {float(location.get("longitude", 0)):.5f}

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # NDVI / NBR
    # --------------------------------------------------------

    ndvi = evidence.get(
        "ndvi",
        {}
    )

    nbr = evidence.get(
        "nbr",
        {}
    )

    ndvi_center = ndvi.get(
        "center"
    )

    nbr_center = nbr.get(
        "center"
    )

    ndvi_mean = ndvi.get(
        "mean"
    )

    nbr_mean = nbr.get(
        "mean"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        if ndvi_center is not None:

            st.metric(
                "NDVI at hotspot",
                f"{float(ndvi_center):.4f}"
            )

        else:

            st.metric(
                "NDVI at hotspot",
                "N/A"
            )

        if ndvi_mean is not None:

            st.caption(
                f"Window mean: "
                f"{float(ndvi_mean):.4f}"
            )

    with metric_col2:

        if nbr_center is not None:

            st.metric(
                "NBR at hotspot",
                f"{float(nbr_center):.4f}"
            )

        else:

            st.metric(
                "NBR at hotspot",
                "N/A"
            )

        if nbr_mean is not None:

            st.caption(
                f"Window mean: "
                f"{float(nbr_mean):.4f}"
            )

    # --------------------------------------------------------
    # RANGE INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "📊 Sentinel spectral statistics"
    ):

        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:

            st.markdown("**NDVI**")

            st.write(
                f"Minimum: "
                f"{float(ndvi.get('min', 0)):.4f}"
            )

            st.write(
                f"Mean: "
                f"{float(ndvi.get('mean', 0)):.4f}"
            )

            st.write(
                f"Maximum: "
                f"{float(ndvi.get('max', 0)):.4f}"
            )

        with stat_col2:

            st.markdown("**NBR**")

            st.write(
                f"Minimum: "
                f"{float(nbr.get('min', 0)):.4f}"
            )

            st.write(
                f"Mean: "
                f"{float(nbr.get('mean', 0)):.4f}"
            )

            st.write(
                f"Maximum: "
                f"{float(nbr.get('max', 0)):.4f}"
            )

    # --------------------------------------------------------
    # SCL
    # --------------------------------------------------------

    st.markdown(
        "### 🧭 Scene Classification"
    )

    scl = evidence.get(
        "scl",
        {}
    )

    center_class = scl.get(
        "center_class",
        "N/A"
    )

    st.write(
        f"**Center SCL class:** {center_class}"
    )

    classes = scl.get(
        "classes",
        {}
    )

    if classes:

        st.write(
            "**SCL classes in analysis window:**"
        )

        scl_columns = st.columns(
            min(len(classes), 4)
        )

        for index, (
            class_id,
            count
        ) in enumerate(
            classes.items()
        ):

            with scl_columns[
                index % len(scl_columns)
            ]:

                st.metric(
                    f"SCL {class_id}",
                    f"{count:,} px"
                )

    # --------------------------------------------------------
    # PROCESSING INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### ⚙️ Processing Information"
    )

    processing_col1, processing_col2 = st.columns(2)

    with processing_col1:

        buffer_m = evidence.get(
            "buffer_m",
            "N/A"
        )

        st.write(
            f"**Analysis buffer:** "
            f"{buffer_m} m"
        )

    with processing_col2:

        array_shape = evidence.get(
            "array_shape",
            {}
        )

        height = array_shape.get(
            "height",
            "N/A"
        )

        width = array_shape.get(
            "width",
            "N/A"
        )

        st.write(
            f"**Processing grid:** "
            f"{width} × {height}"
        )

    # --------------------------------------------------------
    # IMPORTANT NOTE
    # --------------------------------------------------------

    st.info(
        "🛰️ Sentinel-2 provides higher-resolution "
        "contextual surface evidence around the FIRMS "
        "thermal hotspot. It should not be interpreted "
        "as definitive proof of an active fire. "
        "The Sentinel acquisition may also differ "
        "in time from the FIRMS observation."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔥 SIH26162 Fire Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based thermal hotspot classification with '
    'NASA FIRMS and Sentinel-2 contextual evidence'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🔥 Fire Detection"
    )

    st.markdown(
        """
        **SIH26162**

        AI-based detection and classification
        of industrial fires and persistent
        thermal sources.
        """
    )

    st.divider()

    event_id = st.number_input(
        "Event ID",
        min_value=0,
        value=84111,
        step=1
    )

    detect_fire = st.button(
        "🔥 Detect Fire",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.markdown(
        "**Pipeline**"
    )

    st.write(
        "NASA FIRMS → ML Model → "
        "Sentinel-2 Evidence"
    )

    st.caption(
        "FastAPI backend connected"
    )


# ============================================================
# DETECT FIRE
# ============================================================

if detect_fire:

    current_event_id = int(event_id)

    # Clear previous Sentinel result
    st.session_state.sentinel_result = None

    try:

        # ====================================================
        # FASTAPI PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing thermal event..."
        ):

            data = predict_event(
                current_event_id
            )

        # Save result
        st.session_state.event_result = data

        st.session_state.last_event_id = (
            current_event_id
        )

        # ====================================================
        # EXTRACT DATA
        # ====================================================

        event = data["event"]

        location = data["location"]

        prediction = data["prediction"]

        latitude = float(
            location["latitude"]
        )

        longitude = float(
            location["longitude"]
        )

        source = prediction["source"]

        confidence = float(
            prediction["confidence"]
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "🔥 Thermal hotspot successfully analyzed."
        )

        # ====================================================
        # TOP METRICS
        # ====================================================

        st.subheader(
            "📊 AI Classification"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Predicted Source",
                source
            )

        with col2:

            st.metric(
                "Model Score",
                f"{confidence * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Mean FRP",
                f"{float(event.get('mean_frp', 0)):.2f}"
            )

        with col4:

            st.metric(
                "Peak FRP",
                f"{float(event.get('peak_frp', 0)):.2f}"
            )

        # ====================================================
        # LOCATION
        # ====================================================

        st.divider()

        st.subheader(
            "📍 Detected Hotspot Location"
        )

        location_col1, location_col2 = st.columns(2)

        with location_col1:

            st.write(
                f"**Latitude:** "
                f"{latitude:.5f}"
            )

            st.write(
                f"**Longitude:** "
                f"{longitude:.5f}"
            )

        with location_col2:

            st.write(
                f"**Event ID:** "
                f"{current_event_id}"
            )

            st.write(
                f"**Observations:** "
                f"{event.get('observation_count', 'N/A')}"
            )

        # ====================================================
        # FIRMS MAP
        # ====================================================

        st.divider()

        st.subheader(
            "🗺️ FIRMS Hotspot Location"
        )

        fire_map = create_firms_map(
            latitude=latitude,
            longitude=longitude,
            event_id=current_event_id,
            source=source,
            confidence=confidence,
            event=event
        )

        st_folium(
            fire_map,
            width=None,
            height=600,
            returned_objects=[]
        )

        # ====================================================
        # SENTINEL API
        # ====================================================

        st.divider()

        with st.spinner(
            "Loading Sentinel-2 evidence..."
        ):

            sentinel_data = get_sentinel_evidence(
                current_event_id
            )

        st.session_state.sentinel_result = (
            sentinel_data
        )

        display_sentinel_evidence(
            sentinel_data,
            current_event_id
        )

        # ====================================================
        # EVENT DETAILS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Thermal Event Details"
        )

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(
                f"**Start time:** "
                f"{event.get('start_time', 'N/A')}"
            )

            st.write(
                f"**End time:** "
                f"{event.get('end_time', 'N/A')}"
            )

            st.write(
                f"**Observation count:** "
                f"{event.get('observation_count', 'N/A')}"
            )

        with detail_col2:

            st.write(
                f"**Mean FRP:** "
                f"{event.get('mean_frp', 'N/A')}"
            )

            st.write(
                f"**Peak FRP:** "
                f"{event.get('peak_frp', 'N/A')}"
            )

            st.write(
                f"**Persistent:** "
                f"{event.get('persistent', 'N/A')}"
            )

        # ====================================================
        # MAP LINKS
        # ====================================================

        st.divider()

        st.subheader(
            "🌐 Open Location"
        )

        maps = location.get(
            "maps",
            {}
        )

        google_maps = maps.get(
            "google_maps"
        )

        openstreetmap = maps.get(
            "openstreetmap"
        )

        link_col1, link_col2 = st.columns(2)

        with link_col1:

            if google_maps:

                st.link_button(
                    "🌎 Open Google Maps",
                    google_maps,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Google Maps link unavailable."
                )

        with link_col2:

            if openstreetmap:

                st.link_button(
                    "🗺️ OpenStreetMap",
                    openstreetmap,
                    use_container_width=True
                )

            else:

                st.warning(
                    "OpenStreetMap link unavailable."
                )

        # ====================================================
        # METHODOLOGY
        # ====================================================

        st.divider()

        st.subheader(
            "ℹ️ Methodology"
        )

        st.caption(
            "NASA FIRMS provides satellite-based "
            "thermal hotspot detections. A hotspot "
            "coordinate should not be interpreted as "
            "an exact building-level fire location. "
            "Sentinel-2 provides contextual surface "
            "evidence and does not by itself confirm "
            "an active fire."
        )

    # ========================================================
    # FASTAPI CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ FastAPI is not running."
        )

        st.info(
            "Start the backend first:"
        )

        st.code(
            "python -m uvicorn api.main:app --reload",
            language="powershell"
        )

    # ========================================================
    # TIMEOUT ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ Request timed out."
        )

        st.info(
            "The Sentinel-2 processing request can "
            "take longer because JP2 files may be "
            "large. Try again."
        )

    # ========================================================
    # GENERAL REQUEST ERROR
    # ========================================================

    except requests.exceptions.RequestException as e:

        st.error(
            "❌ API request failed."
        )

        st.code(
            str(e)
        )

    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Unexpected error."
        )

        st.exception(e)


# ============================================================
# SHOW LAST RESULT AFTER STREAMLIT RERUN
# ============================================================

elif (
    st.session_state.event_result is not None
    and
    st.session_state.last_event_id == int(event_id)
):

    data = st.session_state.event_result

    event = data.get(
        "event",
        {}
    )

    location = data.get(
        "location",
        {}
    )

    prediction = data.get(
        "prediction",
        {}
    )

    latitude = float(
        location.get(
            "latitude",
            0
        )
    )

    longitude = float(
        location.get(
            "longitude",
            0
        )
    )

    source = prediction.get(
        "source",
        "N/A"
    )

    confidence = float(
        prediction.get(
            "confidence",
            0
        )
    )

    # --------------------------------------------------------
    # LAST RESULT HEADER
    # --------------------------------------------------------

    st.info(
        f"Showing analysis for Event "
        f"{int(event_id)}"
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "📊 AI Classification"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Predicted Source",
            source
        )

    with col2:

        st.metric(
            "Model Score",
            f"{confidence * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Mean FRP",
            f"{float(event.get('mean_frp', 0)):.2f}"
        )

    with col4:

        st.metric(
            "Peak FRP",
            f"{float(event.get('peak_frp', 0)):.2f}"
        )

    # --------------------------------------------------------
    # FIRMS MAP
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🗺️ FIRMS Hotspot Location"
    )

    fire_map = create_firms_map(
        latitude=latitude,
        longitude=longitude,
        event_id=int(event_id),
        source=source,
        confidence=confidence,
        event=event
    )

    st_folium(
        fire_map,
        width=None,
        height=600,
        returned_objects=[]
    )

    # --------------------------------------------------------
    # SENTINEL
    # --------------------------------------------------------

    st.divider()

    if st.session_state.sentinel_result is not None:

        display_sentinel_evidence(
            st.session_state.sentinel_result,
            int(event_id)
        )

    else:

        st.warning(
            "Sentinel-2 evidence has not been loaded yet. "
            "Click Detect Fire."
        )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.info(
        "Enter an Event ID and click "
        "**🔥 Detect Fire** to analyze a thermal event."
    )

    st.markdown(
        """
        ### Pipeline

        **NASA FIRMS**
        → **Thermal Event**
        → **LightGBM Source Classification**
        → **Sentinel-2 Context**
        → **Decision Support**
        """
    )

    st.caption(
        "Prototype dashboard for SIH26162."
    )