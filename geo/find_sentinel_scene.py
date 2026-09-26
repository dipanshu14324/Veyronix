from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

EVENT_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)

EVENT_ID = 84111

SEARCH_DAYS = 5
MAX_CLOUD = 40


# ============================================================
# LOAD ACTUAL EVENT
# ============================================================

print("\nLoading event-level dataset...")

events = pd.read_csv(
    EVENT_FILE,
    low_memory=False
)

event = events[
    events["event_id"] == EVENT_ID
]

if len(event) == 0:
    print(f"Event {EVENT_ID} not found.")
    raise SystemExit

event = event.iloc[0]


latitude = float(event["latitude"])
longitude = float(event["longitude"])
start_time = pd.to_datetime(
    event["start_time"]
)

event_date = start_time.date()


print("\nActual thermal event:")

print("Event ID :", EVENT_ID)
print("Latitude :", latitude)
print("Longitude:", longitude)
print("Start    :", start_time)


# ============================================================
# SEARCH WINDOW
# ============================================================

start_date = (
    start_time - timedelta(days=SEARCH_DAYS)
).strftime("%Y-%m-%dT00:00:00.000Z")

end_date = (
    start_time + timedelta(days=SEARCH_DAYS)
).strftime("%Y-%m-%dT23:59:59.999Z")


# ============================================================
# SMALL AOI AROUND HOTSPOT
# ============================================================

delta = 0.02

min_lon = longitude - delta
max_lon = longitude + delta

min_lat = latitude - delta
max_lat = latitude + delta

aoi = (
    f"POLYGON(("
    f"{min_lon} {min_lat},"
    f"{max_lon} {min_lat},"
    f"{max_lon} {max_lat},"
    f"{min_lon} {max_lat},"
    f"{min_lon} {min_lat}"
    f"))"
)


# ============================================================
# COPERNICUS DATA SPACE
# ============================================================

base_url = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
)


filter_query = (
    "Collection/Name eq 'SENTINEL-2' "

    "and "
    "Attributes/OData.CSC.StringAttribute/"
    "any(att:att/Name eq 'productType' "
    "and att/OData.CSC.StringAttribute/"
    "Value eq 'S2MSI2A') "

    "and "
    f"OData.CSC.Intersects("
    f"area=geography'SRID=4326;"
    f"{aoi}') "

    "and "
    f"ContentDate/Start gt {start_date} "

    "and "
    f"ContentDate/Start lt {end_date} "

    "and "
    "Attributes/OData.CSC.DoubleAttribute/"
    "any(att:att/Name eq 'cloudCover' "
    f"and att/OData.CSC.DoubleAttribute/"
    f"Value le {MAX_CLOUD})"
)


params = {
    "$filter": filter_query,
    "$orderby": "ContentDate/Start desc",
    "$top": "20"
}


print("\nSearching Sentinel-2 L2A scenes...")

response = requests.get(
    base_url,
    params=params,
    timeout=60
)

print(
    "HTTP status:",
    response.status_code
)


if response.status_code != 200:

    print(response.text[:2000])

    raise SystemExit


data = response.json()

products = data.get("value", [])


print(
    f"Scenes found: {len(products)}"
)


# ============================================================
# DISPLAY
# ============================================================

if not products:

    print(
        "\nNo suitable Sentinel-2 L2A scene found."
    )

else:

    print("\nMatching Sentinel-2 scenes:")

    print("=" * 100)

    for product in products:

        print(
            "\nName:",
            product.get("Name")
        )

        print(
            "ID:",
            product.get("Id")
        )

        print(
            "Start:",
            product
            .get("ContentDate", {})
            .get("Start")
        )

        print(
            "S3Path:",
            product.get("S3Path")
        )

        print(
            "Footprint:",
            "Available"
            if product.get("GeoFootprint")
            else "None"
        )


print("\nDone.")