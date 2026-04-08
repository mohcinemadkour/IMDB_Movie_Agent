"""
tests/test_vectorstore.py
--------------------------
Tests for data/vectorstore.py — provider dispatch, FAISS local build/load,
graceful error handling for missing packages.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestProviderDispatch:
    """Test that get_vectorstore() routes to the correct backend."""

    def test_default_is_faiss(self, sample_df, tmp_path):
        with patch.dict(os.environ, {"VECTOR_STORE": "faiss"}, clear=False):
            with patch("data.vectorstore._get_faiss") as mock_faiss:
                mock_faiss.return_value = MagicMock()
                from data.vectorstore import get_vectorstore
                get_vectorstore(sample_df)
                mock_faiss.assert_called_once()

    def test_unknown_provider_falls_back_to_faiss(self, sample_df):
        with patch.dict(os.environ, {"VECTOR_STORE": "invalid_backend"}, clear=False):
            with patch("data.vectorstore._get_faiss") as mock_faiss:
                mock_faiss.return_value = MagicMock()
                from data.vectorstore import get_vectorstore
                get_vectorstore(sample_df)
                mock_faiss.assert_called_once()

    def test_pinecone_provider_routes_correctly(self, sample_df):
        with patch.dict(os.environ, {"VECTOR_STORE": "pinecone"}, clear=False):
            with patch("data.vectorstore._get_pinecone") as mock_pinecone:
                mock_pinecone.return_value = MagicMock()
                from data.vectorstore import get_vectorstore
                get_vectorstore(sample_df)
                mock_pinecone.assert_called_once()

    def test_chroma_provider_routes_correctly(self, sample_df):
        with patch.dict(os.environ, {"VECTOR_STORE": "chroma"}, clear=False):
            with patch("data.vectorstore._get_chroma") as mock_chroma:
                mock_chroma.return_value = MagicMock()
                from data.vectorstore import get_vectorstore
                get_vectorstore(sample_df)
                mock_chroma.assert_called_once()


class TestBuildDocuments:
    """Test the _build_documents helper."""

    def test_document_count_matches_rows(self, sample_df):
        from data.vectorstore import _build_documents
        docs = _build_documents(sample_df)
        assert len(docs) == len(sample_df)

    def test_page_content_is_overview(self, sample_df):
        from data.vectorstore import _build_documents
        docs = _build_documents(sample_df)
        assert docs[0].page_content == sample_df.iloc[0]["Overview"]

    def test_metadata_keys(self, sample_df):
        from data.vectorstore import _build_documents
        docs = _build_documents(sample_df)
        expected_keys = {"Series_Title", "Released_Year", "Genre", "IMDB_Rating",
                         "Meta_score", "Director"}
        for doc in docs:
            assert expected_keys == set(doc.metadata.keys())

    def test_metadata_types(self, sample_df):
        from data.vectorstore import _build_documents
        docs = _build_documents(sample_df)
        for doc in docs:
            assert isinstance(doc.metadata["Series_Title"], str)
            assert isinstance(doc.metadata["Released_Year"], int)
            assert isinstance(doc.metadata["IMDB_Rating"], float)


class TestFaissBackend:
    """Test FAISS build and load-from-disk behaviour without calling OpenAI."""

    def test_faiss_loads_from_disk_when_index_exists(self, sample_df, tmp_path):
        from unittest.mock import MagicMock, patch

        fake_vs = MagicMock()
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")  # simulate non-empty dir

        with patch("data.vectorstore._FAISS_INDEX_PATH", str(index_dir)):
            with patch("langchain_community.vectorstores.FAISS") as mock_faiss_cls:
                mock_faiss_cls.load_local.return_value = fake_vs
                with patch("data.vectorstore._get_embeddings") as mock_emb:
                    mock_emb.return_value = MagicMock()
                    from data.vectorstore import _get_faiss
                    result = _get_faiss(sample_df, force_rebuild=False)

        mock_faiss_cls.load_local.assert_called_once()
        assert result is fake_vs

    def test_faiss_builds_when_index_missing(self, sample_df, tmp_path):
        from unittest.mock import MagicMock, patch

        fake_vs = MagicMock()
        empty_dir = tmp_path / "empty_index"
        # Do NOT create the dir — simulate missing index

        with patch("data.vectorstore._FAISS_INDEX_PATH", str(empty_dir)):
            with patch("langchain_community.vectorstores.FAISS") as mock_faiss_cls:
                mock_faiss_cls.from_documents.return_value = fake_vs
                with patch("data.vectorstore._get_embeddings") as mock_emb:
                    mock_emb.return_value = MagicMock()
                    from data.vectorstore import _get_faiss
                    result = _get_faiss(sample_df, force_rebuild=False)

        mock_faiss_cls.from_documents.assert_called_once()
        assert result is fake_vs

    def test_force_rebuild_skips_disk_load(self, sample_df, tmp_path):
        from unittest.mock import MagicMock, patch

        fake_vs = MagicMock()
        index_dir = tmp_path / "faiss_index"
        index_dir.mkdir()
        (index_dir / "index.faiss").write_bytes(b"fake")

        with patch("data.vectorstore._FAISS_INDEX_PATH", str(index_dir)):
            with patch("langchain_community.vectorstores.FAISS") as mock_faiss_cls:
                mock_faiss_cls.from_documents.return_value = fake_vs
                with patch("data.vectorstore._get_embeddings") as mock_emb:
                    mock_emb.return_value = MagicMock()
                    from data.vectorstore import _get_faiss
                    _get_faiss(sample_df, force_rebuild=True)

        # Should rebuild, not load_local
        mock_faiss_cls.load_local.assert_not_called()
        mock_faiss_cls.from_documents.assert_called_once()


class TestPineconeBackend:
    """Test Pinecone backend error handling (no real API calls)."""

    def test_missing_api_key_raises_value_error(self, sample_df):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PINECONE_API_KEY", None)

            # Patch the import so the test doesn't need pinecone installed
            import sys
            fake_pinecone = MagicMock()
            fake_lc_pinecone = MagicMock()
            with patch.dict(sys.modules, {
                "pinecone": fake_pinecone,
                "langchain_pinecone": fake_lc_pinecone,
            }):
                from data.vectorstore import _get_pinecone
                with pytest.raises(ValueError, match="PINECONE_API_KEY"):
                    _get_pinecone(sample_df)
