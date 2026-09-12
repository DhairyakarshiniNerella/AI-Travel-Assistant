import requests


def get_coordinates(city):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": city,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "accept-language": "en"
    }

    headers = {
        "User-Agent": "TravelMateAI/1.0"
    }

    response = requests.get(url, params=params, headers=headers)

    response.raise_for_status()

    data = response.json()

    if not data:
        return None

    location = data[0]

    return {
        "name": location.get("name") or city,
        "latitude": float(location["lat"]),
        "longitude": float(location["lon"]),
        "country": location.get("address", {}).get("country")
    }