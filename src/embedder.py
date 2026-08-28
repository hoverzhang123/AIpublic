from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Encodes text into fixed-dimensional vector embeddings.

    Wraps the sentence-transformers library to convert text (documents and queries)
    into semantic vector representations. The Embedder uses the "all-MiniLM-L6-v2"
    model, which generates 384-dimensional embeddings optimized for semantic similarity.

    Embeddings are used to populate the Milvus vector database and perform semantic
    search during query time. This component is initialized once at pipeline startup
    and shared across document indexing and query processing.
    """

    def __init__(self) -> None:
        """
        Initialize the Embedder by loading the pre-trained sentence-transformers model.

        Loads the "all-MiniLM-L6-v2" model from the sentence-transformers library.
        This is a lightweight, efficient model optimized for semantic similarity tasks.
        Model loading is done once during initialization for performance.
        """
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of documents into vector embeddings.

        Converts text documents into fixed-dimensional vector representations using
        the sentence-transformers model's encode_document method. Used during the
        build_collection phase to generate embeddings for all indexed documents
        that will be stored in Milvus.

        Args:
            texts (List[str]): List of document sections to encode (e.g., from DataLoader).
                              Each string is one document section.

        Returns:
            List[List[float]]: List of embedding vectors, one per input text.
                              Each vector has dimension 384 (for all-MiniLM-L6-v2).
                              Order matches the input order.

        Example:
            >>> embedder = Embedder()
            >>> docs = ["Milvus is a vector database", "Embeddings enable semantic search"]
            >>> vectors = embedder.encode_documents(docs)
            >>> len(vectors)  # 2
            >>> len(vectors[0])  # 384
        """
        return self._model.encode_document(texts)

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Encode a list of user queries into vector embeddings.

        Converts natural language queries into fixed-dimensional vector representations
        for semantic similarity search against Milvus. Used by RagEngine to embed user
        questions before searching the vector database.

        Args:
            queries (List[str]): List of query strings to encode (e.g., ["How is data stored?"]).
                                Each string is one user query.

        Returns:
            List[List[float]]: List of embedding vectors, one per input query.
                              Each vector has dimension 384 (for all-MiniLM-L6-v2).
                              Order matches the input order.

        Example:
            >>> embedder = Embedder()
            >>> queries = ["What is vector search?"]
            >>> vectors = embedder.encode_queries(queries)
            >>> len(vectors)  # 1
            >>> len(vectors[0])  # 384
        """
        return self._model.encode_query(queries)
