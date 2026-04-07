"""
data/vectorstore.py
-------------------
Builds and persists a FAISS vector index over the Overview (plot) column.

First call builds the index (requires OPENAI_API_KEY for embeddings) and
saves it to data/faiss_index/.  Subsequent calls load from disk.
"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Index lives inside the data/ package folder so it travels with the code.
_INDEX_PATH = str(Path(__file__).parent / "faiss_index")


def _build_documents(df: pd.DataFrame) -> list[Document]:
    """Convert each DataFrame row into a LangChain Document."""
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "Series_Title": str(row["Series_Title"]),
            "Released_Year": int(row["Released_Year"]),
            "Genre": str(row["Genre"]),
            "IMDB_Rating": float(row["IMDB_Rating"]),
            "Meta_score": float(row["Meta_score"]) if pd.notna(row.get("Meta_score")) else None,
            "Director": str(row["Director"]),
        }
        docs.append(Document(page_content=str(row["Overview"]), metadata=metadata))
    return docs


def get_vectorstore(df: pd.DataFrame, force_rebuild: bool = False) -> FAISS:
    """
    Return a FAISS vectorstore over Overview embeddings.

    Loads from disk if the index exists; builds and saves otherwise.
    Set force_rebuild=True to re-embed the entire dataset.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_dir = Path(_INDEX_PATH)

    if not force_rebuild and index_dir.exists() and any(index_dir.iterdir()):
        # allow_dangerous_deserialization is safe here: we wrote this file ourselves.
        return FAISS.load_local(
            _INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

    docs = _build_documents(df)
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(_INDEX_PATH)
    return vectorstore
