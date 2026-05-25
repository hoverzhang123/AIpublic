import argparse
import logging

from .config import Config
from .data_loader import DataLoader
from .embedder import Embedder
from .milvus_store import MilvusStore
from .rag_engine import RagEngine


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def build_collection(config: Config, force_rebuild: bool = False) -> MilvusStore:
    loader = DataLoader(config.faq_path)
    documents = loader.load_documents()
    if not documents:
        raise RuntimeError("No documents were loaded from the FAQ source directory.")

    embedder = Embedder()
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
    args = parser.parse_args()

    config = Config()
    config.validate()
    configure_logging(config.log_level)

    store = build_collection(config=config, force_rebuild=args.rebuild)
    engine = RagEngine(config=config, milvus_store=store, embedder=Embedder())

    logging.info("Running RAG query: %s", args.question)
    answer = engine.query(args.question)
    print("\n=== RAG ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
