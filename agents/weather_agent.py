from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools.weather_tool import weather_tool
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


weather_agent = create_agent(
    model=llm,
    tools=[weather_tool],
    system_prompt="""
    You are a helpful weather assistant.

    When the user asks about the weather of a city,
    use the weather tool.

    The weather tool accepts a city name directly.

    Give the weather information in simple and clear language.
    """
)
