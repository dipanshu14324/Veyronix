import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Map Test",
    layout="wide"
)

st.title("🔥 Fire Map Test")

latitude = 18.6813
longitude = 73.0320

fire_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=15
)

folium.Marker(
    location=[latitude, longitude],
    tooltip="🔥 Fire Hotspot",
    popup="Event ID: 84111",
).add_to(fire_map)

st_folium(
    fire_map,
    width=None,
    height=600
)