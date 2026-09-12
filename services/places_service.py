import os
import requests
from dotenv import load_dotenv

load_dotenv()


def search_places(latitude, longitude, category):
    api_key = os.getenv("GEOAPIFY_API_KEY")

    url = "https://api.geoapify.com/v2/places"

    params = {
        "categories": category,
        "filter": f"circle:{longitude},{latitude},5000",
        "limit": 5,
        "apiKey": api_key
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    return response.json()