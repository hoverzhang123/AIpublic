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
    """
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    collection_name: str = os.getenv("MILVUS_COLLECTION", "my_rag_collection")
    openai_api_key: str = os.getenv("LOCAL_API_KEY") or os.getenv("OPENAI_API_KEY")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
    faq_path: Path = Path(os.getenv("MILVUS_DOCS_PATH", "src/milvus_docs/en/faq"))
    max_retrievals: int = int(os.getenv("MAX_RETRIEVALS", "3"))
    metric_type: str = os.getenv("MILVUS_METRIC", "IP")
    consistency_level: str = os.getenv("MILVUS_CONSISTENCY", "Strong")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

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
