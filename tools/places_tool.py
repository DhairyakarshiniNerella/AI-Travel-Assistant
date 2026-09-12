from langchain_core.tools import tool

from services.geocoding_service import get_coordinates
from services.places_service import search_places


@tool
def places_tool(city: str, category: str):
    """
    Search for nearby restaurants, hotels, and tourist attractions in a city.
    """

    location = get_coordinates(city)

    if location is None:
        return f"Could not find the location: {city}"

    category_map = {
        "restaurant": "catering.restaurant",
        "restaurants": "catering.restaurant",

        "hotel": "accommodation.hotel",
        "hotels": "accommodation.hotel",

        "tourist": "tourism",
        "tourist attraction": "tourism",
        "tourist attractions": "tourism",

        "attraction": "tourism",
        "attractions": "tourism",

        "temple": "tourism",
        "temples": "tourism",

        "monument": "tourism",
        "monuments": "tourism",

        "landmark": "tourism",
        "landmarks": "tourism",
    }

    geoapify_category = category_map.get(category.lower())

    if geoapify_category is None:
        return f"Unsupported category: {category}"

    data = search_places(
        location["latitude"],
        location["longitude"],
        geoapify_category
    )

    results = data.get("features", [])

    if not results:
        return f"No {category} places found in {city}."

    places = []

    for place in results[:5]:

        properties = place.get("properties", {})

        name = properties.get("name", "Unknown")

        # Build a short address instead of using Geoapify's
        # very long "formatted" address.
        address_parts = [
            properties.get("address_line1"),
            properties.get("address_line2"),
            properties.get("city"),
            properties.get("postcode")
        ]

        address = ", ".join(
            str(part)
            for part in address_parts
            if part
        )

        if not address:
            address = f"{city}"

        contact = properties.get("contact") or {}

        phone = contact.get("phone", "Not available")

        places.append({
            "name": name,
            "address": address,
            "phone": phone
        })

    return places