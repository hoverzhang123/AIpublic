from typing import Dict, List

from openai import OpenAI

from .config import Config
from .embedder import Embedder
from .milvus_store import MilvusStore


class RagEngine:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) pipeline.

    Coordinates document retrieval, prompt construction, and LLM-based answer generation.
    Takes a user question, retrieves relevant documents from Milvus, builds a system prompt
    and user prompt with context, and calls an LLM (via OpenRouter) to generate the answer.

    This is the main query-time component; it's initialized once at startup and called for
    each user question.
    """
    def __init__(self, config: Config, milvus_store: MilvusStore, embedder: Embedder) -> None:
        """
        Initialize the RAG engine with dependencies.

        Sets up the OpenAI-compatible client pointing to OpenRouter and stores
        references to the configuration, Milvus store, and embedder for use
        during query processing.

        Args:
            config (Config): Configuration object containing API keys, model names,
                            retrieval limits, and other settings.
            milvus_store (MilvusStore): Initialized Milvus client for vector search.
            embedder (Embedder): Initialized embedder for encoding queries.
        """
        self.config = config
        self.milvus_store = milvus_store
        self.embedder = embedder
        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.llm_base_url,
        )
        self.model_name = self.config.openrouter_model

    def _build_prompt(self, context: str, question: str) -> Dict[str, str]:
        """
        Construct system and user prompts for the LLM.

        Creates a structured prompt pair with a system message instructing the LLM
        to use provided context, and a user message containing the context passages
        and the question to answer.

        Args:
            context (str): Retrieved document passages from Milvus, typically concatenated
                          from multiple search results with newlines.
            question (str): The user's question/query.

        Returns:
            Dict[str, str]: Dictionary with keys "system" and "user" containing the
                           respective prompt strings.

        Example:
            >>> engine = RagEngine(config, store, embedder)
            >>> prompts = engine._build_prompt("Data is stored...", "How is data stored?")
            >>> prompts["system"]  # System instructions
            >>> prompts["user"]  # User prompt with context and question
        """
        system_prompt = """
Human: You are an AI assistant. You can find the answer to the question from the provided context paragraphs.
"""
        user_prompt = f"""
Please use the information fragments enclosed in <context> tags below to answer the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
        return {"system": system_prompt, "user": user_prompt}

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM to generate an answer based on system and user prompts.

        Makes an API call to OpenRouter (using OpenAI-compatible API) with the
        constructed prompts and returns the generated response text.

        Args:
            system_prompt (str): System message instructing the LLM's behavior.
            user_prompt (str): User message containing context and question.

        Returns:
            str: The LLM-generated answer text extracted from the response.

        Raises:
            Possible exceptions from OpenAI client library (APIError, AuthenticationError, etc.)
            if the API call fails.

        Example:
            >>> engine = RagEngine(config, store, embedder)
            >>> answer = engine._generate_response(system_msg, user_msg)
            >>> print(answer)  # LLM-generated answer
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def query(self, question: str) -> str:
        """
        Execute a complete RAG query: retrieve context and generate an answer.

        Orchestrates the full RAG pipeline:
        1. Encodes the question into a vector embedding
        2. Searches Milvus for top-k similar documents
        3. Concatenates retrieved passages into context
        4. Builds system and user prompts with context
        5. Calls the LLM to generate a grounded answer

        This is the main public entry point called by main.py for each user query.

        Args:
            question (str): The user's question (e.g., "How is data stored in Milvus?").

        Returns:
            str: The LLM-generated answer grounded in retrieved context.

        Raises:
            RuntimeError: If no similar documents are found in Milvus for the query.
            Possible exceptions from LLM API calls (see _generate_response).

        Example:
            >>> engine = RagEngine(config, store, embedder)
            >>> answer = engine.query("How is data stored in Milvus?")
            >>> print(answer)
        """
        query_embedding = self.embedder.encode_queries([question])[0]
        matches = self.milvus_store.search(
            query_embedding=query_embedding,
            limit=self.config.max_retrievals,
            metric_type=self.config.metric_type,
        )

        if not matches:
            raise RuntimeError("No similar content found in Milvus.")

        context = "\n".join(match["text"] for match in matches)
        prompts = self._build_prompt(context=context, question=question)
        return self._generate_response(
            system_prompt=prompts["system"], user_prompt=prompts["user"]
        )
