"""
Unit tests for src/milvus_store.py — Milvus vector database operations.

Tests MilvusStore wrapper for collection lifecycle and vector operations.
Uses mocking to avoid Docker/Milvus dependencies.
"""

from unittest.mock import Mock, patch, call

import pytest

from src.milvus_store import MilvusStore


class TestMilvusStoreInitialization:
    """Tests for MilvusStore initialization."""

    def test_milvus_store_init_stores_parameters(self):
        """MilvusStore stores uri and collection_name."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client_class.return_value = Mock()

            store = MilvusStore("http://localhost:19530", "test_collection")

            assert store.collection_name == "test_collection"

    def test_milvus_store_initializes_client(self):
        """MilvusStore initializes MilvusClient with correct uri."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")

            mock_client_class.assert_called_once_with(uri="http://localhost:19530")
            assert store.client == mock_client


class TestListCollections:
    """Tests for MilvusStore.list_collections() method."""

    def test_list_collections_returns_list(self):
        """list_collections() returns list from client."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.list_collections.return_value = ["collection1", "collection2"]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "my_collection")
            result = store.list_collections()

            assert result == ["collection1", "collection2"]

    def test_list_collections_delegates_to_client(self):
        """list_collections() calls client.list_collections()."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "my_collection")
            store.list_collections()

            mock_client.list_collections.assert_called_once()

    def test_list_collections_returns_empty_list(self):
        """list_collections() returns empty list when no collections exist."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.list_collections.return_value = []
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "my_collection")
            result = store.list_collections()

            assert result == []


class TestHasCollection:
    """Tests for MilvusStore.has_collection() method."""

    def test_has_collection_returns_true_when_exists(self):
        """has_collection() returns True when collection exists."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.has_collection.return_value = True
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            result = store.has_collection()

            assert result is True

    def test_has_collection_returns_false_when_missing(self):
        """has_collection() returns False when collection doesn't exist."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.has_collection.return_value = False
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            result = store.has_collection()

            assert result is False

    def test_has_collection_checks_correct_collection(self):
        """has_collection() checks the correct collection name."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "my_collection")
            store.has_collection()

            mock_client.has_collection.assert_called_once_with("my_collection")


class TestCreateCollection:
    """Tests for MilvusStore.create_collection() method."""

    def test_create_collection_calls_client_with_defaults(self):
        """create_collection() calls client with default metric and consistency."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            store.create_collection(dimension=384)

            mock_client.create_collection.assert_called_once_with(
                collection_name="test_collection",
                dimension=384,
                metric_type="IP",
                consistency_level="Strong",
            )

    def test_create_collection_accepts_custom_metric_type(self):
        """create_collection() accepts custom metric_type."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            store.create_collection(dimension=384, metric_type="COSINE")

            args, kwargs = mock_client.create_collection.call_args
            assert kwargs["metric_type"] == "COSINE"

    def test_create_collection_accepts_custom_consistency(self):
        """create_collection() accepts custom consistency_level."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            store.create_collection(dimension=384, consistency_level="Eventually")

            args, kwargs = mock_client.create_collection.call_args
            assert kwargs["consistency_level"] == "Eventually"


class TestDropCollection:
    """Tests for MilvusStore.drop_collection() method."""

    def test_drop_collection_drops_if_exists(self):
        """drop_collection() drops collection if it exists."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.has_collection.return_value = True
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            store.drop_collection()

            mock_client.drop_collection.assert_called_once_with("test_collection")

    def test_drop_collection_skips_if_not_exists(self):
        """drop_collection() skips dropping if collection doesn't exist."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.has_collection.return_value = False
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            store.drop_collection()

            # drop_collection should not be called on client
            mock_client.drop_collection.assert_not_called()


class TestInsert:
    """Tests for MilvusStore.insert() method."""

    def test_insert_with_valid_data(self):
        """insert() adds documents and embeddings to collection."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            docs = ["Doc 1", "Doc 2"]
            embeddings = [[0.1, 0.2], [0.3, 0.4]]

            store.insert(docs, embeddings)

            # Verify client.insert was called
            mock_client.insert.assert_called_once()

            # Verify data structure
            call_args = mock_client.insert.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            data = call_args[1]["data"]
            assert len(data) == 2
            assert data[0]["text"] == "Doc 1"
            assert data[0]["vector"] == [0.1, 0.2]
            assert data[1]["text"] == "Doc 2"

    def test_insert_assigns_sequential_ids(self):
        """insert() assigns sequential IDs to documents."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            docs = ["Doc 1", "Doc 2", "Doc 3"]
            embeddings = [[0.1], [0.2], [0.3]]

            store.insert(docs, embeddings)

            call_args = mock_client.insert.call_args
            data = call_args[1]["data"]
            ids = [item["id"] for item in data]
            assert ids == [0, 1, 2]

    def test_insert_raises_on_length_mismatch(self):
        """insert() raises ValueError if documents and embeddings lengths differ."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client_class.return_value = Mock()

            store = MilvusStore("http://localhost:19530", "test_collection")
            docs = ["Doc 1", "Doc 2"]
            embeddings = [[0.1]]  # Only one embedding

            with pytest.raises(ValueError, match="same length"):
                store.insert(docs, embeddings)

    def test_insert_raises_on_empty_docs_with_embeddings(self):
        """insert() raises ValueError if documents is empty but embeddings provided."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client_class.return_value = Mock()

            store = MilvusStore("http://localhost:19530", "test_collection")

            with pytest.raises(ValueError):
                store.insert([], [[0.1]])

    def test_insert_with_large_vectors(self):
        """insert() handles large embedding dimensions correctly."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            docs = ["Doc 1"]
            large_embedding = [0.1 * i for i in range(384)]  # 384-dim
            embeddings = [large_embedding]

            store.insert(docs, embeddings)

            call_args = mock_client.insert.call_args
            data = call_args[1]["data"]
            assert len(data[0]["vector"]) == 384


class TestSearch:
    """Tests for MilvusStore.search() method."""

    def test_search_returns_correct_format(self):
        """search() returns list of dicts with 'text' and 'distance' keys."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [
                [
                    {"entity": {"text": "Result 1"}, "distance": 0.95},
                    {"entity": {"text": "Result 2"}, "distance": 0.87},
                ]
            ]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1 * i for i in range(384)]
            results = store.search(query_embedding, limit=2)

            assert isinstance(results, list)
            assert len(results) == 2
            assert "text" in results[0]
            assert "distance" in results[0]
            assert results[0]["text"] == "Result 1"
            assert results[0]["distance"] == 0.95

    def test_search_calls_client_with_correct_params(self):
        """search() calls client.search with correct parameters."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [[]]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1 * i for i in range(384)]

            store.search(query_embedding, limit=3, metric_type="COSINE")

            call_args = mock_client.search.call_args
            assert call_args[1]["collection_name"] == "test_collection"
            assert call_args[1]["limit"] == 3
            assert call_args[1]["data"] == [query_embedding]

    def test_search_with_default_metric_type(self):
        """search() uses default metric_type if not specified."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [[]]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1 * i for i in range(384)]

            store.search(query_embedding, limit=3)

            call_args = mock_client.search.call_args
            search_params = call_args[1]["search_params"]
            assert search_params["metric_type"] == "IP"

    def test_search_returns_empty_list_for_no_results(self):
        """search() returns empty list if no results found."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [[]]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1 * i for i in range(384)]
            results = store.search(query_embedding, limit=3)

            assert results == []

    def test_search_wraps_query_embedding(self):
        """search() wraps query_embedding in list for client call."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [[]]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1, 0.2, 0.3]

            store.search(query_embedding, limit=3)

            call_args = mock_client.search.call_args
            # Query embedding should be wrapped in list
            assert call_args[1]["data"] == [[0.1, 0.2, 0.3]]

    def test_search_specifies_output_fields(self):
        """search() specifies 'text' as output field."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [[]]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            query_embedding = [0.1 * i for i in range(384)]

            store.search(query_embedding, limit=3)

            call_args = mock_client.search.call_args
            assert call_args[1]["output_fields"] == ["text"]


class TestSearchEdgeCases:
    """Tests for edge cases in search operations."""

    def test_search_with_single_result(self):
        """search() handles single result correctly."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [
                [{"entity": {"text": "Only result"}, "distance": 0.99}]
            ]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            results = store.search([0.1], limit=1)

            assert len(results) == 1
            assert results[0]["text"] == "Only result"

    def test_search_preserves_result_order(self):
        """search() preserves order of results from client."""
        with patch("src.milvus_store.MilvusClient") as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = [
                [
                    {"entity": {"text": "First"}, "distance": 0.9},
                    {"entity": {"text": "Second"}, "distance": 0.8},
                    {"entity": {"text": "Third"}, "distance": 0.7},
                ]
            ]
            mock_client_class.return_value = mock_client

            store = MilvusStore("http://localhost:19530", "test_collection")
            results = store.search([0.1], limit=3)

            assert results[0]["text"] == "First"
            assert results[1]["text"] == "Second"
            assert results[2]["text"] == "Third"
