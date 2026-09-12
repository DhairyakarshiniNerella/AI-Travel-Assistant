import base64
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

VISION_MODEL = "qwen/qwen3.6-27b"


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def identify_image(image_path):
    """Use a vision-capable LLM to identify the place/subject of an image."""

    image_base64 = _encode_image(image_path)

    extension = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "jpeg" if extension == "jpg" else extension

    llm = ChatGroq(
        model=VISION_MODEL,
        temperature=0,
        max_tokens=500,
        reasoning_effort="none"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Identify the landmark, place, or subject shown in this "
                    "travel photo. Give the name, city/country if "
                    "identifiable, and a 2-3 sentence description useful for "
                    "a travel assistant's knowledge base. If nothing "
                    "identifiable is visible, briefly describe what the "
                    "image shows instead."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{mime};base64,{image_base64}"
                }
            }
        ]
    )

    response = llm.invoke([message])

    return response.content.strip()
