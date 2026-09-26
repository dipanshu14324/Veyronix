import requests

PRODUCT_ID = "b808db89-eb1e-46d9-a2ba-1261f85836ce"

SAFE_NAME = (
    "S2C_MSIL2A_20250131T054121_N0511_R005_T43QBA_20250131T124701.SAFE"
)

GRANULE_NAME = "L2A_T43QBA_A002120_20250131T054332"

url = (
    "https://download.dataspace.copernicus.eu/"
    f"odata/v1/Products({PRODUCT_ID})/"
    f"Nodes({SAFE_NAME})/"
    "Nodes(GRANULE)/"
    f"Nodes({GRANULE_NAME})/"
    "Nodes(IMG_DATA)/"
    "Nodes(R20m)/Nodes"
)

print("\nListing R20m files...")
print(url)

response = requests.get(
    url,
    timeout=60
)

print("\nHTTP status:", response.status_code)

if response.status_code != 200:
    print(response.text[:3000])
    raise SystemExit

data = response.json()
results = data.get("result", [])

print(f"\nFiles found: {len(results)}")

for i, node in enumerate(results, start=1):

    print("\n" + "=" * 80)
    print(f"File #{i}")

    print("Name:")
    print(node.get("Name"))

    print("ID:")
    print(node.get("Id"))

    print("Content length:")
    print(node.get("ContentLength"))

print("\nDone.")