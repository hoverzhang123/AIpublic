"""
Unit tests for src/rag_engine.py - RAG pipeline orchestration.

Tests the RagEngine class for prompt building, LLM generation, and
orchestration of the full RAG query pipeline. Uses mocking to avoid
external LLM API calls.
"""

from unittest.mock import Mock, patch

import pytest

from src.rag_engine import RagEngine


class TestRagEngineInitialization:
    """Tests for RagEngine initialization."""

    def test_rag_engine_init_stores_dependencies(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """RagEngine stores config, milvus_store, and embedder."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = Mock()

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)

            assert engine.config == sample_config
            assert engine.milvus_store == mock_milvus_store
            assert engine.embedder == mock_embedder

    def test_rag_engine_initializes_openai_client(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """RagEngine initializes OpenAI client with correct API key and URL."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_openai_client = Mock()
            mock_openai_class.return_value = mock_openai_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)

            mock_openai_class.assert_called_once_with(
                api_key=sample_config.openai_api_key,
                base_url=sample_config.llm_base_url,
            )
            assert engine.client == mock_openai_client

    def test_rag_engine_stores_model_name(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """RagEngine stores the configured model name."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)

            assert engine.model_name == sample_config.openrouter_model


class TestBuildPrompt:
    """Tests for RagEngine._build_prompt() method."""

    def test_build_prompt_returns_dict_with_system_and_user(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() returns dict with 'system' and 'user' keys."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            prompts = engine._build_prompt(
                context="Test context", question="Test question"
            )

            assert isinstance(prompts, dict)
            assert "system" in prompts
            assert "user" in prompts
            assert isinstance(prompts["system"], str)
            assert isinstance(prompts["user"], str)

    def test_build_prompt_includes_context(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() includes the provided context in user prompt."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            context = "Milvus stores vectors efficiently"
            prompts = engine._build_prompt(
                context=context, question="How does Milvus work?"
            )

            assert context in prompts["user"]

    def test_build_prompt_includes_question(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() includes the provided question in user prompt."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            question = "How is data stored in Milvus?"
            prompts = engine._build_prompt(context="Some context", question=question)

            assert question in prompts["user"]

    def test_build_prompt_system_message_instructs_llm(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() system message instructs LLM to use context."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            prompts = engine._build_prompt(context="Context", question="Question")

            # System message should mention context or provide instructions
            assert (
                "context" in prompts["system"].lower()
                or "information" in prompts["system"].lower()
            )

    def test_build_prompt_with_empty_context(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() handles empty context."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            prompts = engine._build_prompt(context="", question="Question")

            assert prompts["user"] is not None
            assert len(prompts["user"]) > 0

    def test_build_prompt_with_multiline_context(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_build_prompt() preserves multiline context."""
        with patch("src.rag_engine.OpenAI"):
            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            context = "Line 1\nLine 2\nLine 3"
            prompts = engine._build_prompt(context=context, question="Question")

            assert context in prompts["user"]


class TestGenerateResponse:
    """Tests for RagEngine._generate_response() method."""

    def test_generate_response_calls_openai_client(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """_generate_response() calls OpenAI client with correct parameters."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            response = engine._generate_response("System prompt", "User prompt")

            mock_client.chat.completions.create.assert_called_once()

    def test_generate_response_passes_correct_model(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """_generate_response() uses the configured model."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            engine._generate_response("System", "User")

            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == sample_config.openrouter_model

    def test_generate_response_passes_messages_correctly(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """_generate_response() passes system and user messages in correct format."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            system_prompt = "You are helpful"
            user_prompt = "What is Milvus?"
            engine._generate_response(system_prompt, user_prompt)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == system_prompt
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == user_prompt

    def test_generate_response_returns_content(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """_generate_response() returns the LLM-generated content."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            response = engine._generate_response("System", "User")

            assert (
                response
                == "This is the LLM-generated answer based on the context provided."
            )

    def test_generate_response_extracts_correct_message(
        self, sample_config, mock_milvus_store, mock_embedder
    ):
        """_generate_response() correctly extracts message content from response."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            # Setup mock response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            expected_content = "Generated answer from LLM"
            mock_response.choices[0].message.content = expected_content

            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            result = engine._generate_response("System", "User")

            assert result == expected_content


class TestQuery:
    """Tests for RagEngine.query() orchestration method."""

    def test_query_encodes_question(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() encodes the question using embedder."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            question = "How is data stored?"
            engine.query(question)

            # Verify embedder was called with question
            mock_embedder.encode_queries.assert_called_once_with([question])

    def test_query_searches_milvus_with_embedding(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() searches Milvus with the encoded query embedding."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            # Setup embedder to return predictable embedding
            query_embedding = [0.1 * i for i in range(384)]
            mock_embedder.encode_queries.return_value = [query_embedding]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            engine.query("question")

            # Verify Milvus search was called
            mock_milvus_store.search.assert_called_once()

            # Verify correct parameters
            call_args = mock_milvus_store.search.call_args
            assert call_args[1]["query_embedding"] == query_embedding
            assert call_args[1]["limit"] == sample_config.max_retrievals

    def test_query_raises_when_no_matches_found(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() raises RuntimeError when Milvus returns no results."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Setup: Milvus search returns empty list
            mock_milvus_store.search.return_value = []

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)

            with pytest.raises(RuntimeError, match="No similar content found"):
                engine.query("question")

    def test_query_builds_context_from_results(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() concatenates retrieved documents into context."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            # Setup: Milvus returns specific results
            mock_milvus_store.search.return_value = [
                {"text": "Document 1", "distance": 0.9},
                {"text": "Document 2", "distance": 0.8},
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            engine.query("question")

            # Verify LLM was called with context containing both documents
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_message = messages[1]["content"]

            assert "Document 1" in user_message
            assert "Document 2" in user_message

    def test_query_calls_llm_with_context_and_question(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() calls LLM with both context and question."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            mock_milvus_store.search.return_value = [
                {"text": "Some document", "distance": 0.9}
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            question = "What is Milvus?"
            engine.query(question)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            user_message = messages[1]["content"]

            # Both context and question should be in user message
            assert "Some document" in user_message
            assert question in user_message

    def test_query_returns_llm_response(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() returns the LLM-generated answer."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            mock_milvus_store.search.return_value = [
                {"text": "Some document", "distance": 0.9}
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            response = engine.query("question")

            assert (
                response
                == "This is the LLM-generated answer based on the context provided."
            )

    def test_query_uses_correct_metric_type(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() passes config metric_type to Milvus search."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            mock_milvus_store.search.return_value = [
                {"text": "Document", "distance": 0.9}
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            engine.query("question")

            call_args = mock_milvus_store.search.call_args
            assert call_args[1]["metric_type"] == sample_config.metric_type

    def test_query_with_multiple_results(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """query() correctly handles multiple search results."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            # Setup: 3 results from search
            mock_milvus_store.search.return_value = [
                {"text": "Result 1", "distance": 0.95},
                {"text": "Result 2", "distance": 0.87},
                {"text": "Result 3", "distance": 0.76},
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            response = engine.query("question")

            # All results should be in context
            call_args = mock_client.chat.completions.create.call_args
            user_message = call_args[1]["messages"][1]["content"]

            assert "Result 1" in user_message
            assert "Result 2" in user_message
            assert "Result 3" in user_message


class TestRagEngineIntegration:
    """Integration-style tests for full query workflow."""

    def test_full_query_workflow(
        self, sample_config, mock_milvus_store, mock_embedder, mock_openai_response
    ):
        """Full query workflow orchestrates all components correctly."""
        with patch("src.rag_engine.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client

            # Setup realistic returns
            embedding = [0.1 * i for i in range(384)]
            mock_embedder.encode_queries.return_value = [embedding]
            mock_milvus_store.search.return_value = [
                {"text": "Milvus stores vectors", "distance": 0.92},
                {"text": "Using similarity search", "distance": 0.88},
            ]

            engine = RagEngine(sample_config, mock_milvus_store, mock_embedder)
            answer = engine.query("How does Milvus work?")

            # Verify full workflow executed
            assert isinstance(answer, str)
            assert len(answer) > 0
            mock_embedder.encode_queries.assert_called_once()
            mock_milvus_store.search.assert_called_once()
            mock_client.chat.completions.create.assert_called_once()
