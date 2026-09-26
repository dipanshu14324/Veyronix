import os
from pathlib import Path
import requests

PRODUCT_ID = "b808db89-eb1e-46d9-a2ba-1261f85836ce"

SAFE_NAME = (
    "S2C_MSIL2A_20250131T054121_N0511_R005_T43QBA_20250131T124701.SAFE"
)

GRANULE_NAME = "L2A_T43QBA_A002120_20250131T054332"

BASE_URL = "https://download.dataspace.copernicus.eu"

OUTPUT_DIR = Path("data/sentinel/event_84111")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILENAME = "T43QBA_20250131T054121_SCL_20m.jp2"
EXPECTED_SIZE = 1800815


# ============================================================
# AUTHENTICATION
# ============================================================

USERNAME = os.getenv("CDSE_USERNAME")
PASSWORD = os.getenv("CDSE_PASSWORD")

if not USERNAME or not PASSWORD:
    raise RuntimeError("CDSE credentials are not set.")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

response = requests.post(
    TOKEN_URL,
    data={
        "client_id": "cdse-public",
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password",
    },
    timeout=60,
)

response.raise_for_status()

token = response.json()["access_token"]

headers = {
    "Authorization": f"Bearer {token}"
}


# ============================================================
# DOWNLOAD
# ============================================================

output_file = OUTPUT_DIR / FILENAME

if output_file.exists() and output_file.stat().st_size == EXPECTED_SIZE:
    print("SCL already downloaded.")
    raise SystemExit

url = (
    f"{BASE_URL}/odata/v1/"
    f"Products({PRODUCT_ID})/"
    f"Nodes({SAFE_NAME})/"
    f"Nodes(GRANULE)/"
    f"Nodes({GRANULE_NAME})/"
    "Nodes(IMG_DATA)/"
    "Nodes(R20m)/"
    f"Nodes({FILENAME})/$value"
)

print("Downloading SCL...")
print("Expected:", EXPECTED_SIZE, "bytes")

with requests.get(
    url,
    headers=headers,
    stream=True,
    timeout=120
) as r:

    print("HTTP status:", r.status_code)

    r.raise_for_status()

    with open(output_file, "wb") as f:

        for chunk in r.iter_content(chunk_size=1024 * 1024):

            if chunk:
                f.write(chunk)

print("\nSCL downloaded successfully.")
print("Size:", output_file.stat().st_size)