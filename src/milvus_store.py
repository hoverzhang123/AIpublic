from typing import Any, Dict, List

from pymilvus import MilvusClient


class MilvusStore:
    def __init__(self, uri: str, collection_name: str) -> None:
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name

    def list_collections(self) -> List[str]:
        return self.client.list_collections()

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def drop_collection(self) -> None:
        if self.has_collection():
            self.client.drop_collection(self.collection_name)

    def create_collection(
        self,
        dimension: int,
        metric_type: str = "IP",
        consistency_level: str = "Strong",
    ) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type=metric_type,
            consistency_level=consistency_level,
        )

    def insert(self, documents: List[str], embeddings: List[List[float]]) -> None:
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
