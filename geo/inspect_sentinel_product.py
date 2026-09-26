from pathlib import Path
import requests


PRODUCT_ID = (
    "b808db89-eb1e-46d9-a2ba-1261f85836ce"
)

BASE_URL = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
)

url = f"{BASE_URL}({PRODUCT_ID})"

print("\nInspecting Sentinel-2 product...")

response = requests.get(
    url,
    timeout=60
)

print("HTTP status:", response.status_code)

if response.status_code != 200:

    print(response.text[:2000])

    raise SystemExit


product = response.json()

print("\nProduct name:")
print(product.get("Name"))

print("\nProduct ID:")
print(product.get("Id"))

print("\nS3 path:")
print(product.get("S3Path"))

print("\nProduct type:")
print(product.get("ProductType"))

print("\nContent date:")
print(product.get("ContentDate"))

print("\nFootprint available:")
print(
    product.get("GeoFootprint") is not None
)

print("\nDone.")