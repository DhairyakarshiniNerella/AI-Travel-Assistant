from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from tools.weather_tool import weather_tool
from tools.rag_tool import travel_knowledge_tool
from tools.places_tool import places_tool

from dotenv import load_dotenv


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1024,
    reasoning_effort="low",
    model_kwargs={
        "frequency_penalty": 0.8,
        "presence_penalty": 0.4
    }
)


tools = [
    weather_tool,
    places_tool,
    travel_knowledge_tool
]


llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState):

    system_message = """
You are an AI Travel Assistant.

You have access to three tools.

1. Weather tool:
   Use weather_tool when the user asks about current,
   live, or today's weather.

2. Places tool:
   Use places_tool when the user wants to search
   for restaurants, hotels, or tourist attractions
   (including temples, monuments, and landmarks)
   using the places service.

3. Travel knowledge tool:
   Use travel_knowledge_tool when the user asks
   about information contained in the uploaded
   travel documents.

IMPORTANT TOOL SELECTION RULES:

- If the user asks "according to the travel guide",
  "according to the uploaded document", "from the
  travel guide", or otherwise explicitly refers to the
  uploaded document, ALWAYS use travel_knowledge_tool.

- For every other question about real-world hotels,
  restaurants, temples, monuments, landmarks, or tourist
  attractions in a specific city, ALWAYS try places_tool
  first, even if the topic also happens to be covered in
  an uploaded document. Only fall back to
  travel_knowledge_tool or general knowledge if places_tool
  returns no results or an unsupported-category error.

FACTUAL ACCURACY RULES:

- Never invent specific, checkable details: exact
  addresses, phone numbers, URLs, booking links, prices,
  ratings, distances, or travel times. Those must come
  from a tool result or the uploaded document; if you
  don't have them, omit them or say they're unavailable
  instead of making them up.

- Well-known general facts (e.g. the names of famous
  landmarks) may be answered from your own knowledge when
  no tool has better data. If a tool's results look wrong
  or irrelevant for the question, ignore them and answer
  from general knowledge instead of retrying the tool or
  forcing the tool data to fit.

- When listing multiple items, each item must be unique.
  Never repeat the same place, name, or description.

- If the user asks for a specific number of items (e.g.
  "list 50 temples") but you only have a smaller number
  of genuinely distinct, real items available, list only
  that smaller number and say plainly that this is all
  you found. Never keep generating items you don't
  actually know just to reach the requested count.

- Do not explain your internal processing, and do not
  mention duplicate detection, formatting problems, tool
  behavior, or corrections in the final answer.

Answer the user clearly and simply.
"""

    messages = [
        {
            "role": "system",
            "content": system_message
        }
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


graph = StateGraph(MessagesState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    tools_condition
)

graph.add_edge("tools", "agent")


travel_agent = graph.compile()