from langchain_core.tools import tool
from services.geocoding_service import get_coordinates
from services.weather_service import get_weather


@tool
def weather_tool(city: str):
    """
    Get the current weather for a city.
    """

    location = get_coordinates(city)

    if location is None:
        return f"Could not find the location: {city}"

    weather = get_weather(
        location["latitude"],
        location["longitude"]
    )

    current = weather["current"]

    return {
        "city": location["name"],
        "country": location["country"],
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "apparent_temperature": current["apparent_temperature"],
        "weather_code": current["weather_code"],
        "wind_speed": current["wind_speed_10m"]
    }