from pathlib import Path
from datetime import timedelta
import requests
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ml_ready"
    / "phase1_event_level_ml_dataset.csv"
)

SENTINEL_DIR = PROJECT_ROOT / "data" / "sentinel"


# ============================================================
# COPERNICUS STAC API
# ============================================================

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"


# ============================================================
# GET FIRMS EVENT
# ============================================================

def get_event(event_id):
    """
    Read FIRMS event information from the ML dataset.
    """

    if not EVENT_DATASET.exists():
        raise FileNotFoundError(
            f"Event dataset not found:\n{EVENT_DATASET}"
        )

    df = pd.read_csv(EVENT_DATASET)

    event = df[df["event_id"] == int(event_id)]

    if event.empty:
        raise ValueError(
            f"Event ID {event_id} was not found."
        )

    return event.iloc[0]


# ============================================================
# SEARCH SENTINEL-2
# ============================================================

def search_sentinel(
    latitude,
    longitude,
    event_date,
    days_before=5,
    days_after=5,
    max_cloud_cover=60
):
    """
    Search Sentinel-2 L2A scenes around a FIRMS event.

    Searches:
        event_date - days_before
        through
        event_date + days_after
    """

    start_date = (
        pd.Timestamp(event_date)
        - timedelta(days=days_before)
    )

    end_date = (
        pd.Timestamp(event_date)
        + timedelta(days=days_after)
    )

    datetime_range = (
        f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}/"
        f"{end_date.strftime('%Y-%m-%dT23:59:59Z')}"
    )

    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": datetime_range,
        "intersects": {
            "type": "Point",
            "coordinates": [
                float(longitude),
                float(latitude)
            ]
        },
        "limit": 100,
        "query": {
            "eo:cloud_cover": {
                "lte": max_cloud_cover
            }
        }
    }

    response = requests.post(
        STAC_URL,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    features = result.get("features", [])

    return features


# ============================================================
# RANK RESULTS
# ============================================================

def rank_scenes(
    scenes,
    latitude,
    longitude,
    event_date
):
    """
    Rank Sentinel scenes.

    Preference:
    1. Closest acquisition date
    2. Lower cloud cover
    """

    ranked = []

    event_date = pd.Timestamp(event_date)

    for scene in scenes:

        properties = scene.get("properties", {})

        scene_date = properties.get(
            "datetime"
        )

        cloud_cover = properties.get(
            "eo:cloud_cover",
            999
        )

        if scene_date is None:
            continue

        scene_datetime = pd.Timestamp(scene_date)

        date_difference = abs(
            (scene_datetime.tz_localize(None)
             - event_date).total_seconds()
        )

        ranked.append({
            "id": scene.get("id"),
            "datetime": scene_date,
            "cloud_cover": cloud_cover,
            "date_difference_days":
                date_difference / 86400,
            "geometry": scene.get("geometry"),
            "properties": properties,
            "assets": scene.get("assets", {})
        })

    ranked.sort(
        key=lambda x: (
            x["date_difference_days"],
            x["cloud_cover"]
        )
    )

    return ranked


# ============================================================
# MAIN
# ============================================================

def find_best_sentinel_scene(event_id):

    print("=" * 60)
    print("SENTINEL-2 SEARCH")
    print("=" * 60)

    event = get_event(event_id)

    latitude = float(event["latitude"])
    longitude = float(event["longitude"])
    event_date = event["acq_date"]

    print(f"Event ID : {event_id}")
    print(f"Latitude : {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Date     : {event_date}")

    print()
    print("Searching Copernicus Sentinel-2 L2A...")

    scenes = search_sentinel(
        latitude=latitude,
        longitude=longitude,
        event_date=event_date,
        days_before=5,
        days_after=5,
        max_cloud_cover=60
    )

    print(
        f"Scenes found: {len(scenes)}"
    )

    if not scenes:
        print()
        print(
            "❌ No Sentinel-2 scene found "
            "within the search criteria."
        )
        return None

    ranked = rank_scenes(
        scenes,
        latitude,
        longitude,
        event_date
    )

    print()
    print("AVAILABLE SCENES")
    print("-" * 60)

    for i, scene in enumerate(ranked[:10]):

        print(
            f"{i + 1}. "
            f"{scene['id']}"
        )

        print(
            f"   Date: "
            f"{scene['datetime']}"
        )

        print(
            f"   Cloud: "
            f"{scene['cloud_cover']}%"
        )

        print(
            f"   Difference: "
            f"{scene['date_difference_days']:.2f} days"
        )

        print()

    best = ranked[0]

    print("=" * 60)
    print("BEST SENTINEL-2 SCENE")
    print("=" * 60)

    print(
        f"ID: {best['id']}"
    )

    print(
        f"Date: {best['datetime']}"
    )

    print(
        f"Cloud cover: "
        f"{best['cloud_cover']}%"
    )
    print()
    print("=" * 60)
    print("BEST SCENE ASSETS")
    print("=" * 60)

    for name, asset in best.get("assets", {}).items():
        print()
        print("ASSET:", name)
        print("HREF:", asset.get("href"))

    return best


if __name__ == "__main__":

    # Change this when testing another event
    EVENT_ID = 84118

    find_best_sentinel_scene(EVENT_ID)