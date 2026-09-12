import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)

from langchain_core.documents import Document

from rag.image_vision import identify_image


def load_documents(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    # TXT
    if extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    # PDF
    elif extension == ".pdf":

        loader = PyPDFLoader(file_path)

    # DOCX
    elif extension == ".docx":

        loader = Docx2txtLoader(file_path)

    # Excel
    elif extension in [".xlsx", ".xls"]:

        loader = UnstructuredExcelLoader(
            file_path,
            mode="elements"
        )

    # IMAGE
    elif extension in [".jpg", ".jpeg", ".png"]:

        # -----------------------------
        # Identify image content with a vision LLM
        # -----------------------------
        vision_description = identify_image(file_path)

        # -----------------------------
        # Create LangChain Document
        # -----------------------------
        return [
            Document(
                page_content=vision_description,
                metadata={
                    "source": file_path,
                    "type": "image",
                    "vision_identified": bool(vision_description)
                }
            )
        ]

    # Unsupported file
    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # Load normal documents
    return loader.load()
