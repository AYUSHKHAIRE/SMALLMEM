from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid4
import numpy as np
from config.logger_config import logger

class QdrantClientWrapper:
    """
    A lightweight wrapper around QdrantClient for storing and searching vector embeddings.
    Works with both local Qdrant and remote cloud instances.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: str = None):
        """
        Initialize the Qdrant client connection.
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        logger.info(f"Connected to Qdrant at {url}")

    def create_collection(self, collection_name: str, vector_size: int):
        """
        Creates a new collection if it doesn't already exist.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created new collection: {collection_name}")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def recreate_collection(self, collection_name: str, vector_size: int):
        """
        Deletes and recreates a collection. Use carefully!
        """
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Recreated collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            return e

    def add_vector_embeddings(self, collection_name: str, embeddings: list, payloads: list = None):
        """
        Upserts embeddings (vectors) into Qdrant with optional payloads.
        """
        if payloads is None:
            payloads = [{} for _ in embeddings]

        embeddings = np.array(embeddings, dtype=np.float32).tolist()

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload,
            )
            for embedding, payload in zip(embeddings, payloads)
        ]

        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Added {len(points)} vectors to '{collection_name}' collection.")

    def search_vectors(self, collection_name: str, query_vector: list, top_k: int = 5):
        """
        Searches for the most similar vectors in the given collection.
        Returns a list of dicts containing id, score, and payload.
        """
        query_vector = np.array(query_vector, dtype=np.float32).tolist()
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        formatted = [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]
        return formatted

    def count_points(self, collection_name: str) -> int:
        """
        Returns the total number of vectors in the collection.
        """
        count = self.client.count(collection_name=collection_name, exact=True).count
        return count

    def delete_collection(self, collection_name: str):
        """
        Deletes a collection permanently.
        """
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
