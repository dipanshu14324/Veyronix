import os
import requests

USERNAME = os.getenv("CDSE_USERNAME")
PASSWORD = os.getenv("CDSE_PASSWORD")

if not USERNAME or not PASSWORD:
    raise RuntimeError(
        "CDSE_USERNAME/CDSE_PASSWORD are not set in this PowerShell session."
    )

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

data = {
    "client_id": "cdse-public",
    "username": USERNAME,
    "password": PASSWORD,
    "grant_type": "password",
}

print("Requesting CDSE access token...")

response = requests.post(
    TOKEN_URL,
    data=data,
    timeout=60
)

print("HTTP status:", response.status_code)

if response.status_code != 200:
    print("\nAuthentication failed.")
    print(response.text)
    raise SystemExit(1)

token_data = response.json()

access_token = token_data.get("access_token")

if not access_token:
    print("No access token returned.")
    raise SystemExit(1)

print("\nAuthentication successful.")
print("Access token received:", bool(access_token))
print("Token type:", token_data.get("token_type"))
print("Expires in:", token_data.get("expires_in"), "seconds")

# ------------------------------------------------------------
# Test authenticated access to the B04 file
# ------------------------------------------------------------

PRODUCT_ID = "b808db89-eb1e-46d9-a2ba-1261f85836ce"

SAFE_NAME = (
    "S2C_MSIL2A_20250131T054121_N0511_R005_T43QBA_20250131T124701.SAFE"
)

GRANULE_NAME = "L2A_T43QBA_A002120_20250131T054332"

FILE_NAME = "T43QBA_20250131T054121_B04_10m.jp2"

url = (
    "https://download.dataspace.copernicus.eu/"
    f"odata/v1/Products({PRODUCT_ID})/"
    f"Nodes({SAFE_NAME})/"
    "Nodes(GRANULE)/"
    f"Nodes({GRANULE_NAME})/"
    "Nodes(IMG_DATA)/"
    "Nodes(R10m)/"
    f"Nodes({FILE_NAME})/$value"
)

headers = {
    "Authorization": f"Bearer {access_token}",
    "Range": "bytes=0-1023",
}

print("\nTesting authenticated B04 access...")

r = requests.get(
    url,
    headers=headers,
    stream=True,
    timeout=60
)

print("HTTP status:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("Content-Length:", r.headers.get("Content-Length"))
print("Content-Range:", r.headers.get("Content-Range"))
print("Accept-Ranges:", r.headers.get("Accept-Ranges"))

if r.status_code in (200, 206):
    print("\nSUCCESS: authenticated access to B04 works.")
else:
    print("\nB04 access failed.")
    print(r.text[:500])

r.close()