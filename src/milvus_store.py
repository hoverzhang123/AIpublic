from typing import Any, Dict, List

from pymilvus import MilvusClient


class MilvusStore:
    """
    Client wrapper for Milvus vector database operations.

    Provides a high-level interface to Milvus for collection lifecycle management
    (create, drop, list) and vector operations (insert, search). All operations target
    a single collection specified by collection_name.

    The MilvusStore is initialized once at pipeline startup and shared by the
    build_collection and RagEngine components for indexing and retrieval.
    """
    def __init__(self, uri: str, collection_name: str) -> None:
        """
        Initialize the Milvus client connection.

        Args:
            uri (str): Connection URI for the Milvus server (e.g., "http://localhost:19530").
                      Typically read from MILVUS_URI environment variable.
            collection_name (str): Name of the Milvus collection to operate on
                                  (e.g., "my_rag_collection"). Typically read from
                                  MILVUS_COLLECTION environment variable.
        """
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name

    def list_collections(self) -> List[str]:
        """
        List all collections in the Milvus server.

        Returns:
            List[str]: Names of all collections currently stored in Milvus.
                      Empty list if no collections exist.

        Example:
            >>> store = MilvusStore("http://localhost:19530", "my_rag_collection")
            >>> collections = store.list_collections()
            >>> "my_rag_collection" in collections
        """
        return self.client.list_collections()

    def has_collection(self) -> bool:
        """
        Check if the target collection exists in Milvus.

        Returns:
            bool: True if the collection (specified in __init__) exists, False otherwise.

        Example:
            >>> store = MilvusStore("http://localhost:19530", "my_rag_collection")
            >>> if not store.has_collection():
            ...     store.create_collection(dimension=384)
        """
        return self.client.has_collection(self.collection_name)

    def drop_collection(self) -> None:
        """
        Drop (delete) the target collection and all its data from Milvus.

        Safely drops the collection only if it exists. This is a destructive operation
        used during --rebuild to clear existing data before re-indexing.

        Called by build_collection() when --rebuild flag is passed.
        """
        if self.has_collection():
            self.client.drop_collection(self.collection_name)

    def create_collection(
        self,
        dimension: int,
        metric_type: str = "IP",
        consistency_level: str = "Strong",
    ) -> None:
        """
        Create a new collection in Milvus for storing vectors and documents.

        Creates a collection schema with fields for id, vector embeddings, and text content.
        Called during the build_collection phase if the collection does not already exist.

        Args:
            dimension (int): Dimensionality of the embedding vectors. Typically 384 for
                            the all-MiniLM-L6-v2 model. Must match the embedding model
                            output dimension.
            metric_type (str, optional): Distance metric for similarity search.
                                        Default "IP" (inner product).
                                        Other options: "L2" (Euclidean), "COSINE" (cosine similarity).
            consistency_level (str, optional): Milvus consistency level for reads.
                                             Default "Strong". Options: "Strong", "Session",
                                             "Bounded", "Eventually".

        Example:
            >>> store = MilvusStore("http://localhost:19530", "my_rag_collection")
            >>> store.create_collection(dimension=384, metric_type="IP")
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level,
        )

    def insert(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """
        Insert documents and their embeddings into the collection.

        Batches documents with their corresponding embeddings into Milvus, assigning
        each entry a sequential ID and storing both the vector and raw text for later retrieval.
        Called during the build_collection phase after embeddings are generated.

        Args:
            documents (List[str]): List of document sections to insert (e.g., from DataLoader).
            embeddings (List[List[float]]): Corresponding embedding vectors for each document.
                                           Must be same length as documents list.
                                           Each embedding is a list of floats (e.g., 384-dim).

        Raises:
            ValueError: If documents and embeddings lists have different lengths.

        Example:
            >>> store = MilvusStore("http://localhost:19530", "my_rag_collection")
            >>> docs = ["Document 1", "Document 2"]
            >>> vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            >>> store.insert(docs, vectors)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length.")

        data = [
            {"id": idx, "vector": embeddings[idx], "text": documents[idx]}
            for idx in range(len(documents))
        ]

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        metric_type: str = "IP",
    ) -> List[Dict[str, Any]]:
        """
        Search for the top-k most similar documents to a query embedding.

        Performs vector similarity search in Milvus using the provided query embedding
        and returns the top matching documents with their similarity scores.
        Called by RagEngine.query() to retrieve context for LLM generation.

        Args:
            query_embedding (List[float]): Query vector (e.g., from Embedder.encode_queries).
                                          Dimension must match the collection's embedding dimension.
            limit (int): Number of top results to return (e.g., 3 for max_retrievals).
            metric_type (str, optional): Distance metric for the search. Default "IP" (inner product).
                                        Should match the metric used when creating the collection.

        Returns:
            List[Dict[str, Any]]: List of top-k matches, each containing:
                - "text" (str): The document section content
                - "distance" (float): Similarity score (higher is more similar for IP metric)
                Each result is sorted by distance in descending order.

        Example:
            >>> store = MilvusStore("http://localhost:19530", "my_rag_collection")
            >>> query_vec = [0.1, 0.2, ...]  # 384-dim vector
            >>> results = store.search(query_vec, limit=3)
            >>> results[0]["text"]  # Most similar document
            >>> results[0]["distance"]  # Similarity score
        """
        search_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            search_params={"metric_type": metric_type, "params": {}},
            output_fields=["text"],
        )

        hits = []
        for hit in search_results[0]:
            hits.append(
                {
                    "text": hit["entity"]["text"],
                    "distance": hit["distance"],
                }
            )

        return hits
