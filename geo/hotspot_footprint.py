from math import cos, radians


def create_hotspot_area(latitude, longitude, size_m=375):
    """
    Create an approximate VIIRS detection footprint.

    latitude, longitude:
        FIRMS hotspot center

    size_m:
        Nominal VIIRS pixel size.
        Default = 375 m
    """

    # Half of the nominal pixel size
    half = size_m / 2

    # Approximate metres per degree
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * cos(radians(latitude))

    lat_offset = half / meters_per_degree_lat
    lon_offset = half / meters_per_degree_lon

    min_lat = latitude - lat_offset
    max_lat = latitude + lat_offset

    min_lon = longitude - lon_offset
    max_lon = longitude + lon_offset

    polygon = [
        [min_lat, min_lon],
        [min_lat, max_lon],
        [max_lat, max_lon],
        [max_lat, min_lon],
        [min_lat, min_lon],
    ]

    return {
        "center": {
            "latitude": latitude,
            "longitude": longitude
        },
        "size_m": size_m,
        "polygon": polygon
    }