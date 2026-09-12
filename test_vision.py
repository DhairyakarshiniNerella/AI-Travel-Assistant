import os
import base64

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


image_path = "data/documents/tajmahal.jpeg"

image_base64 = encode_image(image_path)


llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=500
)
    


message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                "Identify the landmark or tourist place "
                "shown in this image. "
                "Give the most likely place name, city, "
                "and a short explanation."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]
)


response = llm.invoke([message])

print("\nVISION RESULT:\n")
print(response.content)