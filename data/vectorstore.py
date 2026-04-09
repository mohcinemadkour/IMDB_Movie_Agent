"""
data/vectorstore.py
-------------------
Provider-agnostic vector store for IMDB Overview (plot) embeddings.

Backend is selected via the VECTOR_STORE environment variable:

  faiss    (default) — local FAISS flat file at data/faiss_index/
                       Committed to the repo; no external service needed.
  pinecone           — Pinecone serverless (free tier).
                       Requires: PINECONE_API_KEY
                       Optional: PINECONE_INDEX_NAME (default: imdb-movies)
  chroma             — ChromaDB.
                       Local persistent (default, stored in CHROMA_PERSIST_DIR)
                       or remote HTTP when CHROMA_HOST is set.
                       Optional: CHROMA_HOST, CHROMA_PORT (default 8000),
                                 CHROMA_COLLECTION (default: imdb-overviews),
                                 CHROMA_PERSIST_DIR (default: .chroma)

All backends return a LangChain VectorStore that exposes .similarity_search().
Indexes are populated lazily on first access; subsequent starts skip re-embedding.
Use force_rebuild=True to wipe and re-embed from scratch.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

_LOG = logging.getLogger(__name__)

# FAISS index path (local fallback / default backend)
_FAISS_INDEX_PATH = str(Path(__file__).parent / "faiss_index")

# Embedding model shared by all backends
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536  # output dimension for text-embedding-3-small


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=_EMBEDDING_MODEL)


# ── FAISS backend (local flat-file, default) ──────────────────────────────────

def _get_faiss(df: pd.DataFrame, force_rebuild: bool = False):
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu and langchain-community are required for the FAISS backend.\n"
            "Install: pip install faiss-cpu langchain-community"
        ) from exc

    embeddings = _get_embeddings()
    index_dir = Path(_FAISS_INDEX_PATH)

    if not force_rebuild and index_dir.exists() and any(index_dir.iterdir()):
        _LOG.info("Loading FAISS index from %s", _FAISS_INDEX_PATH)
        # allow_dangerous_deserialization is safe: we wrote this file ourselves.
        return FAISS.load_local(
            _FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

    _LOG.info("Building FAISS index (force_rebuild=%s) …", force_rebuild)
    docs = _build_documents(df)
    vs = FAISS.from_documents(docs, embeddings)
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(_FAISS_INDEX_PATH)
        _LOG.info("FAISS index saved to %s", _FAISS_INDEX_PATH)
    except PermissionError:
        _LOG.warning(
            "PermissionError saving FAISS index to %s — running in-memory only.",
            _FAISS_INDEX_PATH,
        )
    return vs


# ── Pinecone backend (managed, serverless) ────────────────────────────────────

def _get_pinecone(df: pd.DataFrame, force_rebuild: bool = False):
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise ImportError(
            "pinecone and langchain-pinecone are required for the Pinecone backend.\n"
            "Install: pip install pinecone langchain-pinecone"
        ) from exc

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY must be set in your environment to use the Pinecone backend."
        )

    index_name = os.environ.get("PINECONE_INDEX_NAME", "imdb-movies")
    embeddings = _get_embeddings()
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        _LOG.info("Creating Pinecone serverless index '%s' …", index_name)
        pc.create_index(
            name=index_name,
            dimension=_EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        _LOG.info("Pinecone index '%s' already exists.", index_name)

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    if force_rebuild or total_vectors == 0:
        _LOG.info(
            "Upserting %d documents into Pinecone index '%s' (current vectors: %d) …",
            len(df), index_name, total_vectors,
        )
        docs = _build_documents(df)
        PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
        _LOG.info("Pinecone upsert complete.")
    else:
        _LOG.info(
            "Pinecone index '%s' already has %d vectors; skipping upsert "
            "(set force_rebuild=True to re-embed).",
            index_name, total_vectors,
        )

    return PineconeVectorStore(index=index, embedding=embeddings)


# ── ChromaDB backend (local persistent or remote HTTP) ────────────────────────

def _get_chroma(df: pd.DataFrame, force_rebuild: bool = False):
    try:
        import chromadb
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise ImportError(
            "chromadb and langchain-chroma are required for the ChromaDB backend.\n"
            "Install: pip install chromadb langchain-chroma"
        ) from exc

    embeddings = _get_embeddings()
    collection_name = os.environ.get("CHROMA_COLLECTION", "imdb-overviews")
    chroma_host = os.environ.get("CHROMA_HOST", "")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))

    if chroma_host:
        _LOG.info("Connecting to remote ChromaDB at %s:%d …", chroma_host, chroma_port)
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
        _LOG.info("Using local ChromaDB at '%s' …", persist_dir)
        client = chromadb.PersistentClient(path=persist_dir)

    # Probe whether the collection already has documents
    try:
        existing_count = client.get_collection(collection_name).count()
    except Exception:
        existing_count = 0

    if force_rebuild or existing_count == 0:
        if force_rebuild and existing_count > 0:
            _LOG.info("force_rebuild: deleting ChromaDB collection '%s' …", collection_name)
            client.delete_collection(collection_name)
        _LOG.info(
            "Building ChromaDB collection '%s' with %d documents …",
            collection_name, len(df),
        )
        docs = _build_documents(df)
        vs = Chroma.from_documents(
            docs, embeddings, client=client, collection_name=collection_name
        )
        _LOG.info("ChromaDB collection '%s' built successfully.", collection_name)
    else:
        _LOG.info(
            "ChromaDB collection '%s' has %d documents; skipping rebuild.",
            collection_name, existing_count,
        )
        vs = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    return vs


# ── Public API ────────────────────────────────────────────────────────────────

def get_vectorstore(df: pd.DataFrame, force_rebuild: bool = False):
    """Return a LangChain VectorStore over the Overview (plot) embeddings.

    The backend is selected via the VECTOR_STORE environment variable:
      faiss    (default) — local FAISS flat file
      pinecone           — Pinecone serverless
      chroma             — ChromaDB (local persistent or remote HTTP)

    All backends support .similarity_search() and require no changes to callers.
    Indexes are populated lazily; re-embedding only happens when the index is
    empty or when force_rebuild=True is passed.
    """
    provider = os.environ.get("VECTOR_STORE", "faiss").lower()
    _LOG.info("Vector store provider: %s", provider)

    if provider == "pinecone":
        return _get_pinecone(df, force_rebuild=force_rebuild)

    if provider == "chroma":
        return _get_chroma(df, force_rebuild=force_rebuild)

    if provider != "faiss":
        _LOG.warning("Unknown VECTOR_STORE='%s'; falling back to faiss.", provider)

    return _get_faiss(df, force_rebuild=force_rebuild)
