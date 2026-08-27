import argparse
import logging

from .config import Config
from .data_loader import DataLoader
from .embedder import Embedder
from .milvus_store import MilvusStore
from .rag_engine import RagEngine


def configure_logging(level: str) -> None:
    """
    Configure Python logging with timestamps and severity levels.

    Sets up the root logger with a consistent format including timestamp,
    log level, and message. Called once at application startup.

    Args:
        level (str): Logging level as a string (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
                    Typically read from LOG_LEVEL environment variable or Config.log_level.

    Example:
        >>> configure_logging("INFO")
        >>> logging.info("Application started")  # Logs with timestamp
    """
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def build_collection(config: Config, embedder: Embedder, : bool = False) -> MilvusStore:
    """
    Load documents, generate embeddings, and populate Milvus collection.

    Orchestrates the index-building pipeline:
    1. Loads all Markdown documents from MILVUS_DOCS_PATH
    2. Generates embeddings using the Embedder
    3. Creates a Milvus collection if it doesn't exist (or drops if --rebuild)
    4. Inserts documents and embeddings into Milvus

    If the collection already exists and force_rebuild is False, the function logs
    a message and skips re-indexing. If force_rebuild is True, the existing collection
    is dropped before creating a new one.

    This function is called once during main() and returns the initialized MilvusStore
    for use by RagEngine.

    Args:
        config (Config): Configuration object with paths, collection name, dimension settings.
        embedder (Embedder): Initialized embedder to generate vectors for documents.
        force_rebuild (bool, optional): If True, drops existing collection before rebuild.
                                       Default False. Set to True when --rebuild flag is passed.

    Returns:
        MilvusStore: Initialized MilvusStore instance connected to the populated collection,
                    ready for search operations.

    Raises:
        RuntimeError: If no documents are loaded from the FAQ directory.
        Possible exceptions from DataLoader, Embedder, or MilvusStore on I/O or connection errors.

    Example:
        >>> config = Config()
        >>> embedder = Embedder()
        >>> store = build_collection(config, embedder, force_rebuild=False)
        >>> # store is now ready for searching
    """
    loader = DataLoader(config.faq_path)
    documents = loader.load_documents()
    if not documents:
        raise RuntimeError("No documents were loaded from the FAQ source directory.")

    embeddings = embedder.encode_documents(documents)

    store = MilvusStore(uri=config.milvus_uri, collection_name=config.collection_name)

    if force_rebuild and store.has_collection():
        logging.info("Dropping existing Milvus collection: %s", config.collection_name)
        store.drop_collection()

    if not store.has_collection():
        logging.info(
            "Creating Milvus collection %s with dimension %s",
            config.collection_name,
            len(embeddings[0]),
        )
        store.create_collection(
            dimension=len(embeddings[0]),
            metric_type=config.metric_type,
            consistency_level=config.consistency_level,
        )
        logging.info("Inserting %d documents into Milvus collection", len(documents))
        store.insert(documents=documents, embeddings=embeddings)
    else:
        logging.info(
            "Milvus collection already exists: %s. Use --rebuild to recreate it.",
            config.collection_name,
        )

    return store


def main() -> None:
    """
    CLI entry point for the RAG pipeline application.

    Parses command-line arguments, initializes the configuration, sets up logging,
    builds/loads the Milvus index, and executes a RAG query with the provided question.

    Command-line Arguments:
        --question (str): The query question to answer (default: "How is data stored in milvus?")
        --rebuild (flag): If set, drops and rebuilds the Milvus collection before querying.
        --log-level (str): Override logging level (choices: DEBUG, INFO, WARNING, ERROR, CRITICAL).
                          Default uses LOG_LEVEL environment variable (default: INFO).

    Environment Variables (see Config):
        DEEPSEEK_API_KEY / OPENAI_API_KEY: API key for LLM generation (required)
        MILVUS_URI: Milvus server URI (default: http://localhost:19530)
        MILVUS_COLLECTION: Collection name (default: my_rag_collection)
        MILVUS_DOCS_PATH: FAQ source directory (default: src/milvus_docs/en/faq)
        LOG_LEVEL: Python logging level (default: INFO)
        Other settings: See Config class

    Workflow:
        1. Parse CLI arguments
        2. Load and validate Config from environment
        3. Configure logging
        4. Initialize Embedder
        5. Build/load Milvus collection (skip if exists, unless --rebuild)
        6. Initialize RagEngine
        7. Execute query and print answer

    Example:
        >>> # From command line:
        >>> # python -m src.main --question "How is data stored?"
        >>> # python -m src.main --question "What is Milvus?" --rebuild
        >>> # python -m src.main --log-level CRITICAL --question "How is data stored?"

    Raises:
        Possible exceptions from Config.validate(), DataLoader, or RagEngine.query() on errors.
    """
    parser = argparse.ArgumentParser(
        description="Build a RAG index with Milvus and DeepSeek, then answer a query."
    )
    parser.add_argument(
        "--question",
        default="How is data stored in milvus?",
        help="The query text to search and answer.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Milvus collection before querying.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set logging level (default: use LOG_LEVEL environment variable, or INFO).",
    )
    args = parser.parse_args()

    config = Config()
    config.validate()
    # Use CLI --log-level if provided, otherwise fall back to LOG_LEVEL env var
    log_level = args.log_level if args.log_level else config.log_level
    configure_logging(log_level)

    embedder = Embedder()
    store = build_collection(config=config, embedder=embedder, force_rebuild=args.rebuild)
    engine = RagEngine(config=config, milvus_store=store, embedder=embedder)

    logging.info("Running RAG query: %s", args.question)
    answer = engine.query(args.question)
    print("\n=== RAG ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
