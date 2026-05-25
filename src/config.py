import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    collection_name: str = os.getenv("MILVUS_COLLECTION", "my_rag_collection")
    openai_api_key: str = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")
    faq_path: Path = Path(os.getenv("MILVUS_DOCS_PATH", "src/milvus_docs/en/faq"))
    max_retrievals: int = int(os.getenv("MAX_RETRIEVALS", "3"))
    metric_type: str = os.getenv("MILVUS_METRIC", "IP")
    consistency_level: str = os.getenv("MILVUS_CONSISTENCY", "Strong")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "Missing DeepSeek / OpenAI API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
            )
        if not self.faq_path.exists() or not self.faq_path.is_dir():
            raise FileNotFoundError(
                f"FAQ source directory does not exist: {self.faq_path}"
            )
