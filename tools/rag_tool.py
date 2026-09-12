from langchain_core.tools import tool
from rag.retriever import get_retriever


retriever = get_retriever()


@tool
def travel_knowledge_tool(query: str):
    """
    Search the uploaded travel document for relevant information.
    Use this tool when the user asks about information contained
    in the uploaded travel document.
    """

    results = retriever.invoke(query)

    if not results:
        return {
            "answer": "No relevant information found in the uploaded document.",
            "sources": []
        }

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    sources = []

    for result in results:

        metadata = result.metadata

        page = metadata.get("page")

        if isinstance(page, int):
            page = page + 1

        sources.append({
            "source": metadata.get("source", "Unknown"),
            "page": page
        })

    return {
        "answer": context,
        "sources": sources
    }