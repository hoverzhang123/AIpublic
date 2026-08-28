"""
pytest configuration and shared fixtures for RAG pipeline tests.

This module provides reusable fixtures that are automatically discovered by pytest
and available to all tests in the suite. Fixtures include sample data, mock objects,
and temporary resources.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add the repo root to Python path so 'src' package can be imported
# This is needed because pytest runs from the repo root, but without this,
# relative imports like 'from src.config import Config' would fail
repo_root = Path(__file__).parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@pytest.fixture
def sample_markdown_single_section():
    """Sample markdown content without headings."""
    return "This is a single section without any markdown headings."


@pytest.fixture
def sample_markdown_multiple_sections():
    """Sample markdown content with multiple level-1 headings."""
    return """# Section 1
This is the content of section 1.
It spans multiple lines.

# Section 2
This is the content of section 2.

# Section 3
Final section with minimal content."""


@pytest.fixture
def sample_markdown_mixed_levels():
    """Sample markdown with mixed heading levels (should split only on # )."""
    return """# Main Section
Some content here.

## Subsection (not split)
Subsection content is kept with parent.

# Another Main Section
More content."""


@pytest.fixture
def sample_documents():
    """List of sample document sections as would be loaded by DataLoader."""
    return [
        "Milvus is a vector database optimized for similarity search.",
        "Embeddings enable semantic search in large-scale data.",
        "The RAG pipeline combines retrieval and generation.",
        "DeepSeek LLM is used for answer generation.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample 384-dimensional embedding vectors (one per document)."""
    # Create 4 vectors of 384 dimensions each with unique patterns
    return [
        [0.1 * i for i in range(384)],  # Vector 1: incremental values
        [0.2 * (384 - i) for i in range(384)],  # Vector 2: decreasing values
        [0.15 * i * (i % 2) for i in range(384)],  # Vector 3: alternating
        [0.05 * (i + 1) for i in range(384)],  # Vector 4: small increments
    ]


@pytest.fixture
def sample_embedding_single():
    """Single 384-dimensional embedding vector."""
    return [0.1 * i for i in range(384)]


@pytest.fixture
def mock_milvus_client():
    """Mock MilvusClient for testing MilvusStore without Docker."""
    mock = Mock()
    mock.list_collections.return_value = ["existing_collection"]
    mock.has_collection.return_value = True
    mock.create_collection.return_value = None
    mock.drop_collection.return_value = None
    mock.insert.return_value = None
    mock.search.return_value = [
        [
            {"entity": {"text": "Result 1"}, "distance": 0.95},
            {"entity": {"text": "Result 2"}, "distance": 0.87},
            {"entity": {"text": "Result 3"}, "distance": 0.76},
        ]
    ]
    return mock


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI API."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = (
        "This is the LLM-generated answer based on the context provided."
    )
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client for testing RagEngine without API calls."""
    mock = Mock()
    mock.chat = Mock()
    mock.chat.completions = Mock()
    mock.chat.completions.create.return_value = mock_openai_response
    return mock


@pytest.fixture
def mock_embedder():
    """Mock Embedder for testing without loading sentence-transformers model."""
    mock = Mock()
    # Return vectors of correct dimension (384)
    mock.encode_documents.return_value = [
        [0.1 * i for i in range(384)] for _ in range(3)
    ]
    mock.encode_queries.return_value = [[0.1 * i for i in range(384)]]
    return mock


@pytest.fixture
def mock_milvus_store():
    """Mock MilvusStore for testing RagEngine and main orchestration."""
    mock = Mock()
    mock.has_collection.return_value = True
    mock.create_collection.return_value = None
    mock.insert.return_value = None
    mock.search.return_value = [
        {"text": "Retrieved document 1 related to the query", "distance": 0.92},
        {"text": "Retrieved document 2 with relevant information", "distance": 0.88},
        {"text": "Retrieved document 3 somewhat relevant", "distance": 0.75},
    ]
    return mock


@pytest.fixture
def sample_config(tmp_path):
    """Valid Config instance for testing with temporary FAQ directory."""
    # Create a temporary FAQ directory with at least one markdown file
    faq_dir = tmp_path / "faq"
    faq_dir.mkdir()
    (faq_dir / "test.md").write_text("# Test Document\nContent here.")

    # Set environment variables for config
    os.environ["OPENAI_API_KEY"] = "sk-test-key-12345"
    os.environ["MILVUS_URI"] = "http://localhost:19530"
    os.environ["MILVUS_COLLECTION"] = "test_collection"
    os.environ["MILVUS_DOCS_PATH"] = str(faq_dir)
    os.environ["MAX_RETRIEVALS"] = "3"

    # Import here to pick up environment variables
    from src.config import Config

    config = Config()

    yield config

    # Cleanup
    for key in [
        "OPENAI_API_KEY",
        "MILVUS_URI",
        "MILVUS_COLLECTION",
        "MILVUS_DOCS_PATH",
        "MAX_RETRIEVALS",
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def temp_markdown_dir(tmp_path):
    """Temporary directory with sample markdown files for DataLoader testing."""
    # Create main directory
    main_dir = tmp_path / "docs"
    main_dir.mkdir()

    # Create subdirectory
    sub_dir = main_dir / "subfolder"
    sub_dir.mkdir()

    # Create markdown files
    (main_dir / "file1.md").write_text(
        "# Document 1\nContent 1\n# Document 2\nContent 2"
    )
    (sub_dir / "file2.md").write_text("# SubDoc A\nSubcontent A")
    (main_dir / "empty.md").write_text("")

    return main_dir


@pytest.fixture
def autouse_clear_env(monkeypatch):
    """Auto-use fixture that clears test-related environment variables between tests."""
    yield
    # Cleanup after test
    test_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LOCAL_API_KEY", "MILVUS_URI"]
    for key in test_keys:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    """Reset singleton instances between tests to avoid state leakage."""
    yield
    # Clear any cached imports or singletons if needed
    # (In this project, components are dependency-injected, so this is minimal)
