import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

@dataclass(frozen=True)
class Config:
    """
    Centralized configuration management for the RAG pipeline.

    Manages all configuration parameters for the RAG system, reading values from
    environment variables with sensible defaults. The Config object is immutable
    (frozen=True) to prevent accidental mutations after initialization.

    All pipeline components receive a Config instance via dependency injection,
    ensuring consistent configuration across the application.

    Note: Initialize via Config() which reads env vars at instance creation time,
    allowing tests to properly mock environment variables via monkeypatch.
    """
    milvus_uri: str
    collection_name: str
    openai_api_key: str
    openrouter_model: str
    faq_path: Path
    max_retrievals: int
    metric_type: str
    consistency_level: str
    log_level: str
    llm_base_url: str

    def __init__(
        self,
        milvus_uri: str = None,
        collection_name: str = None,
        openai_api_key: str = None,
        openrouter_model: str = None,
        faq_path: Path = None,
        max_retrievals: int = None,
        metric_type: str = None,
        consistency_level: str = None,
        log_level: str = None,
        llm_base_url: str = None,
    ):
        """
        Initialize Config with environment variable defaults.

        All parameters are optional and will read from os.getenv() at initialization time
        (not module import time), allowing proper test mocking.
        """
        object.__setattr__(self, "milvus_uri", milvus_uri or os.getenv("MILVUS_URI", "http://localhost:19530"))
        object.__setattr__(self, "collection_name", collection_name or os.getenv("MILVUS_COLLECTION", "my_rag_collection"))
        object.__setattr__(self, "openai_api_key", openai_api_key or os.getenv("LOCAL_API_KEY") or os.getenv("OPENAI_API_KEY"))
        object.__setattr__(self, "openrouter_model", openrouter_model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free"))
        object.__setattr__(self, "faq_path", faq_path or Path(os.getenv("MILVUS_DOCS_PATH", "src/milvus_docs/en/faq")))
        object.__setattr__(self, "max_retrievals", max_retrievals or int(os.getenv("MAX_RETRIEVALS", "3")))
        object.__setattr__(self, "metric_type", metric_type or os.getenv("MILVUS_METRIC", "IP"))
        object.__setattr__(self, "consistency_level", consistency_level or os.getenv("MILVUS_CONSISTENCY", "Strong"))
        object.__setattr__(self, "log_level", log_level or os.getenv("LOG_LEVEL", "INFO"))
        object.__setattr__(self, "llm_base_url", llm_base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"))

    def validate(self) -> None:
        """
        Validate critical configuration parameters.

        Ensures that all required configuration values are set and valid. This should be
        called once at application startup before initializing other components.

        Raises:
            ValueError: If DEEPSEEK_API_KEY or OPENAI_API_KEY is not set.
            FileNotFoundError: If the FAQ source directory (MILVUS_DOCS_PATH) does not exist
                             or is not a directory.

        Example:
            >>> config = Config()
            >>> config.validate()  # Raises ValueError if API key is missing
        """
        if not self.openai_api_key:
            raise ValueError(
                "Missing DeepSeek / OpenAI API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
            )
        if not self.faq_path.exists() or not self.faq_path.is_dir():
            raise FileNotFoundError(
                f"FAQ source directory does not exist: {self.faq_path}"
            )
