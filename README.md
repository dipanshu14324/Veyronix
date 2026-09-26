# VEYRONIX AI

AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources using NASA FIRMS, OSM and Sentinel-2 satellite evidence.

## SIH 2026

Problem Statement: SIH26162

## Features

- NASA FIRMS thermal hotspot processing
- Thermal event generation
- Historical contextual features
- Anomaly detection
- LightGBM source classification
- OpenStreetMap contextual evidence
- Sentinel-2 satellite evidence
- FastAPI backend
- Streamlit command-center dashboard

## Source Classes

- Industrial
- Agriculture / Biomass
- Forest / Natural
- Waste / Other

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, LightGBM, FastAPI, Uvicorn, Streamlit, GeoPandas, Rasterio, Shapely, PyProj, Folium, NASA FIRMS, VIIRS, Sentinel-2 and OpenStreetMap.

## Run Locally

Backend:

    python -m uvicorn api.main:app --reload

Dashboard:

    streamlit run dashboard/app.py

Swagger:

    http://127.0.0.1:8000/docs

## Deployment

Streamlit Community Cloud
        |
        v
FastAPI on Render
        |
        +-- LightGBM
        +-- FIRMS
        +-- Sentinel-2
        +-- Geospatial Processing

## Security

Do not commit API keys, CDSE credentials, .env files, Sentinel data or raw datasets.

Use environment variables for secrets.

## Important Note

FIRMS provides satellite-detected thermal hotspots. A hotspot coordinate should not automatically be interpreted as an exact building-level fire location.

VEYRONIX combines thermal, historical, geospatial and satellite evidence to support investigation.

## Status

VEYRONIX is an SIH 2026 prototype for thermal-event detection, source attribution and evidence-based fire-event investigation.
