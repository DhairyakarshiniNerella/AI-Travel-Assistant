from langchain_chroma import Chroma

from rag.embeddings import get_embeddings


def get_retriever():

    # Load the same embedding model
    embeddings = get_embeddings()

    # Load existing ChromaDB
    vectorstore = Chroma(
        collection_name="travel_knowledge",
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever