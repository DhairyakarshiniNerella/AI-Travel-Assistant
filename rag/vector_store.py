from langchain_chroma import Chroma

from rag.document_loader import load_documents
from rag.text_splitter import split_documents
from rag.embeddings import get_embeddings


def create_vectorstore(file_path):

    # 1. Load documents
    documents = load_documents(file_path)

    print(f"Loaded {len(documents)} documents")

    # 2. Split documents into chunks
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError(
            "No readable text could be extracted from this file."
        )

    # 3. Get embedding model
    embeddings = get_embeddings()

    # 4. ChromaDB directory
    vectorstore_path = "vectorstore"

    # 5. Connect to existing ChromaDB
    vectorstore = Chroma(
        collection_name="travel_knowledge",
        embedding_function=embeddings,
        persist_directory=vectorstore_path
    )

    # 6. Add new documents to existing database
    vectorstore.add_documents(chunks)

    print("Documents added successfully to ChromaDB!")
    print("Collection: travel_knowledge")

    return vectorstore, documents