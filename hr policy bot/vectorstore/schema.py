# vectorstore/schema.py

from qdrant_client.http import models
from vectorstore.qdrant_client import get_qdrant_client
from llm.embedding_config import get_embeddings
import config.settings as settings


def create_collection():

    client = get_qdrant_client()

    embeddings = get_embeddings()
    test_vector = embeddings.embed_query("test")
    vector_size = len(test_vector)

    # Recreate collection
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    # 🔹 Create payload index for content_hash
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="content_hash",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print(f"Collection '{settings.QDRANT_COLLECTION}' created with payload index.")