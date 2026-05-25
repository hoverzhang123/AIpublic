from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode_document(texts)

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        return self._model.encode_query(queries)
