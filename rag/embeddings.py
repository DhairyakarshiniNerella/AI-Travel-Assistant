import os

# The embedding model is already cached locally after first use, so skip
# HuggingFace Hub's online freshness checks. Without this, a flaky/offline
# connection causes several minutes of retries (5 attempts per config file)
# before falling back to the same cached files anyway. Must run before
# huggingface_hub is imported (directly or via langchain_huggingface) since
# it reads this env var once into a module-level constant.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "token": os.getenv("HF_TOKEN")
        }
    )

    return embeddings