# retriever/vector_retriever.py

from typing import List, Dict

from qdrant_client.http import models

import config.settings as settings
from vectorstore.qdrant_client import get_qdrant_client
from llm.embedding_config import get_embeddings


class VectorRetriever:
    """
    Azure Embedding + Qdrant similarity search retriever.
    
    """
    
    def __init__(self):
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self.collection_name = settings.QDRANT_COLLECTION
        self.top_k = settings.TOP_K

    def retrieve(
    self,
    query: str,
    top_k: int = None,
    metadata_filter: Dict = None
) -> List[Dict]:

        if top_k is None:
            top_k = self.top_k

        # Embed query
        query_vector = self.embeddings.embed_query(query)

        # Optional metadata filtering
        query_filter = None

        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            query_filter = models.Filter(must=conditions)

        # Updated Qdrant API
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points

        results = []

        for hit in search_result:
            results.append({
                "content": hit.payload.get("content"),
                "metadata": hit.payload,
                "score": hit.score
            })

        return results