import os
import requests
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_ID = 84118

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sentinel"
    / f"event_{EVENT_ID}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_ID = "d4b6f2f1-abaa-4dda-bcba-014e63556a5d"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

BASE_URL = (
    "https://download.dataspace.copernicus.eu/"
    "odata/v1"
)


# ============================================================
# AUTHENTICATION
# ============================================================

def get_token():

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "CDSE_USERNAME or CDSE_PASSWORD is missing."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public"
        },
        timeout=60
    )

    response.raise_for_status()

    return response.json()["access_token"]


# ============================================================
# REQUEST JSON
# ============================================================

def get_json(token, url):

    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}"
        },
        timeout=60
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FIND REQUIRED FILES
# ============================================================

TARGETS = {
    "B04_10m": "B04_10m.jp2",
    "B08_10m": "B08_10m.jp2",
    "B12_20m": "B12_20m.jp2",
    "SCL_20m": "SCL_20m.jp2",
}


def find_files(token, nodes_url, found):

    data = get_json(token, nodes_url)

    for node in data.get("result", []):

        name = node.get("Name", "")

        # ----------------------------------------------------
        # FILE?
        # ----------------------------------------------------

        if name.endswith(".jp2"):

            for key, target in TARGETS.items():

                if name.endswith(target):

                    print()
                    print("FOUND:", key)
                    print("NAME :", name)

                    # OData file download URL
                    file_url = (
                        nodes_url.rsplit("/Nodes", 1)[0]
                        + f"/Nodes({node['Id']})/$value"
                    )

                    found[key] = {
                        "name": name,
                        "url": file_url
                    }

                    print("URL  :", file_url)

        # ----------------------------------------------------
        # DIRECTORY?
        # ----------------------------------------------------

        children = node.get("Nodes")

        if children and "uri" in children:

            child_url = children["uri"]

            find_files(
                token,
                child_url,
                found
            )


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(token, key, info):

    output_file = OUTPUT_DIR / info["name"]

    print()
    print("=" * 70)
    print("DOWNLOADING:", key)
    print("=" * 70)

    print("Source:")
    print(info["url"])

    print("Destination:")
    print(output_file)

    headers = {
        "Authorization": f"Bearer {token}"
    }

    with requests.get(
        info["url"],
        headers=headers,
        stream=True,
        timeout=120
    ) as response:

        print("HTTP STATUS:", response.status_code)

        response.raise_for_status()

        total = 0

        with open(output_file, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(chunk)

                    total += len(chunk)

                    print(
                        f"\rDownloaded: "
                        f"{total / (1024 * 1024):.2f} MB",
                        end=""
                    )

    print()
    print("Saved successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COPERNICUS SENTINEL-2 DYNAMIC DOWNLOADER")
    print("=" * 70)

    print("Event ID :", EVENT_ID)
    print("Product  :", PRODUCT_ID)
    print("Output   :", OUTPUT_DIR)

    print()
    print("Authenticating...")

    token = get_token()

    print("Authentication successful.")

    # --------------------------------------------------------
    # START PRODUCT TREE
    # --------------------------------------------------------

    root_url = (
        f"{BASE_URL}/Products({PRODUCT_ID})/Nodes"
    )

    found = {}

    print()
    print("Searching product node tree...")
    print("This may take a little time.")

    find_files(
        token,
        root_url,
        found
    )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for target in TARGETS:

        if target in found:
            print("✓", target)
        else:
            print("✗", target)

    missing = [
        x for x in TARGETS
        if x not in found
    ]

    if missing:

        print()
        print("Missing:")
        for item in missing:
            print(" -", item)

        raise RuntimeError(
            "Required Sentinel-2 files were not found."
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    for key in TARGETS:

        download_file(
            token,
            key,
            found[key]
        )

    print()
    print("=" * 70)
    print("ALL REQUIRED FILES DOWNLOADED")
    print("=" * 70)

    for key in TARGETS:

        path = OUTPUT_DIR / found[key]["name"]

        print(
            f"{key}: "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )


if __name__ == "__main__":
    main()