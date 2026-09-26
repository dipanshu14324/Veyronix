import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="VIIRS Footprint Test",
    layout="wide"
)

st.title("🔥 VIIRS Fire Detection Footprint")


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = (
    "data/processed_data/"
    "phase1_viirs_footprints.csv"
)

df = pd.read_csv(DATA_FILE)


# ============================================================
# EVENT SELECTION
# ============================================================

index = st.number_input(
    "Select observation row",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

row = df.iloc[int(index)]


# ============================================================
# DETAILS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latitude",
    f"{row['latitude']:.5f}"
)

col2.metric(
    "Longitude",
    f"{row['longitude']:.5f}"
)

col3.metric(
    "Scan",
    f"{row['scan']:.2f} km"
)

col4.metric(
    "Track",
    f"{row['track']:.2f} km"
)


st.info(
    "The rectangle represents an estimated VIIRS "
    "detection footprint, not an exact fire boundary."
)


# ============================================================
# MAP
# ============================================================

latitude = row["latitude"]
longitude = row["longitude"]

fire_map = folium.Map(
    location=[
        latitude,
        longitude
    ],
    zoom_start=15,
    control_scale=True
)


# ============================================================
# HOTSPOT CENTER
# ============================================================

folium.Marker(
    location=[
        latitude,
        longitude
    ],
    tooltip="🔥 FIRMS Hotspot",
    popup=(
        f"Latitude: {latitude:.5f}<br>"
        f"Longitude: {longitude:.5f}<br>"
        f"Scan: {row['scan']:.2f} km<br>"
        f"Track: {row['track']:.2f} km"
    ),
).add_to(fire_map)


# ============================================================
# FOOTPRINT
# ============================================================

bounds = [
    [
        row["min_lat"],
        row["min_lon"]
    ],
    [
        row["max_lat"],
        row["max_lon"]
    ]
]


folium.Rectangle(
    bounds=bounds,
    tooltip="Estimated VIIRS Detection Footprint",
    popup=(
        f"Estimated footprint<br>"
        f"Scan: {row['scan']:.2f} km<br>"
        f"Track: {row['track']:.2f} km<br>"
        f"Area: {row['footprint_area_km2']:.3f} km²"
    ),
    fill=True,
    fill_opacity=0.25,
    weight=2
).add_to(fire_map)


# ============================================================
# DISPLAY
# ============================================================

st_folium(
    fire_map,
    width=None,
    height=650,
    returned_objects=[]
)