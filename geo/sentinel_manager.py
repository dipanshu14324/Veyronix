from pathlib import Path
import json


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTINEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sentinel"
)


# ============================================================
# REQUIRED SENTINEL FILES
# ============================================================

REQUIRED_BANDS = {
    "B04": "B04_10m",
    "B08": "B08_10m",
    "B12": "B12_20m",
    "SCL": "SCL_20m",
}


# ============================================================
# EVENT DIRECTORY
# ============================================================

def get_event_directory(event_id):
    """
    Return the Sentinel directory for an event.
    """

    return (
        SENTINEL_ROOT
        / f"event_{event_id}"
    )


# ============================================================
# FIND BAND FILE
# ============================================================

def find_band_file(event_id, band_name):
    """
    Find a Sentinel JP2 file for a specific event.
    """

    event_dir = get_event_directory(event_id)

    if not event_dir.exists():
        return None

    matches = list(
        event_dir.glob(
            f"*_{band_name}.jp2"
        )
    )

    if not matches:
        return None

    return matches[0]


# ============================================================
# CHECK SENTINEL AVAILABILITY
# ============================================================

def check_sentinel_availability(event_id):
    """
    Check whether all required Sentinel files
    are already available locally.
    """

    event_dir = get_event_directory(event_id)

    result = {
        "event_id": int(event_id),
        "available": False,
        "source": "none",
        "directory": str(event_dir),
        "files": {}
    }

    # --------------------------------------------------------
    # EVENT DIRECTORY
    # --------------------------------------------------------

    if not event_dir.exists():

        return result

    # --------------------------------------------------------
    # CHECK EACH BAND
    # --------------------------------------------------------

    all_available = True

    for band, filename_suffix in REQUIRED_BANDS.items():

        file_path = find_band_file(
            event_id,
            filename_suffix
        )

        exists = (
            file_path is not None
            and file_path.exists()
        )

        result["files"][band] = {
            "available": exists,
            "path": (
                str(file_path)
                if exists
                else None
            )
        }

        if not exists:
            all_available = False

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    result["available"] = all_available

    if all_available:
        result["source"] = "local_cache"

    return result


# ============================================================
# GET SENTINEL FILES
# ============================================================

def get_sentinel_files(event_id):
    """
    Return paths of all locally available
    Sentinel files for an event.
    """

    availability = check_sentinel_availability(
        event_id
    )

    if not availability["available"]:

        raise FileNotFoundError(
            f"Complete Sentinel data is not "
            f"available for Event {event_id}"
        )

    files = {}

    for band in REQUIRED_BANDS:

        files[band] = availability["files"][band]["path"]

    return files


# ============================================================
# SAVE METADATA
# ============================================================

def save_sentinel_metadata(
    event_id,
    metadata
):
    """
    Save Sentinel scene metadata locally.
    """

    event_dir = get_event_directory(
        event_id
    )

    event_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        event_dir
        / "sentinel_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return metadata_file


# ============================================================
# LOAD METADATA
# ============================================================

def load_sentinel_metadata(event_id):
    """
    Load Sentinel metadata if available.
    """

    metadata_file = (
        get_event_directory(event_id)
        / "sentinel_metadata.json"
    )

    if not metadata_file.exists():
        return None

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GET EVENT STATUS
# ============================================================

def get_sentinel_status(event_id):
    """
    Return complete Sentinel status for an event.
    """

    availability = check_sentinel_availability(
        event_id
    )

    metadata = load_sentinel_metadata(
        event_id
    )

    return {
        "event_id": int(event_id),
        "available": availability["available"],
        "source": availability["source"],
        "directory": availability["directory"],
        "files": availability["files"],
        "metadata": metadata
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    event_id = 84118

    print()
    print("=" * 60)
    print("SENTINEL MANAGER TEST")
    print("=" * 60)

    status = get_sentinel_status(
        event_id
    )

    print(
        json.dumps(
            status,
            indent=4
        )
    )

    print("=" * 60)