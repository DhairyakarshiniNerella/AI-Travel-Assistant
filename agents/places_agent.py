from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools.places_tool import places_tool
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


places_agent = create_agent(
    model=llm,
    tools=[places_tool],
    system_prompt="""
    You are a helpful travel places assistant.

    When the user asks to find nearby places,
    use the places tool.

    The places tool requires:
    - latitude
    - longitude
    - category

    Give the results in simple and clear language.
    """
)