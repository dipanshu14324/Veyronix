import requests

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

print("\nTesting B04 range request...")
print(url)

headers = {
    "Range": "bytes=0-1023"
}

response = requests.get(
    url,
    headers=headers,
    stream=True,
    timeout=60
)

print("\nHTTP status:", response.status_code)

print("\nContent type:")
print(response.headers.get("Content-Type"))

print("\nContent length:")
print(response.headers.get("Content-Length"))

print("\nContent range:")
print(response.headers.get("Content-Range"))

print("\nAccept ranges:")
print(response.headers.get("Accept-Ranges"))

print("\nFinal URL:")
print(response.url)

response.close()

print("\nDone.")