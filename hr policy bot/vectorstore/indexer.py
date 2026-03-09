# vectorstore/indexer.py

import uuid
from typing import List, Dict

from qdrant_client.http import models

import config.settings as settings
from vectorstore.qdrant_client import get_qdrant_client
from llm.embedding_config import get_embeddings


class AzureQdrantIndexer:
   

    def __init__(self):
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self.collection_name = settings.QDRANT_COLLECTION

    # ---------------------------------------------------------
    # Check if content hash already exists (Duplicate Protection)
    # ---------------------------------------------------------
    def is_duplicate(self, content_hash: str) -> bool:

        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash)
                    )
                ]
            ),
            limit=1
        )

        points = result[0]
        return len(points) > 0

    # ---------------------------------------------------------
    # Batch Index Documents
    # ---------------------------------------------------------
    def index_documents(self, documents: List[Dict]):

        if not documents:
            print("No documents received for indexing.")
            return

        texts = []
        payloads = []

        # Filter duplicates before embedding (cost optimization)
        for doc in documents:

            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})

            if not content:
                continue

            content_hash = metadata.get("content_hash")

            if not content_hash:
                print("Skipping document without content_hash.")
                continue

            if self.is_duplicate(content_hash):
                continue
            
            enriched_text = f"""
            Document: {metadata.get('doc_name', '')}
            Chapter: {metadata.get('chapter', '')}
            Heading: {metadata.get('heading', '')}
            Subheading: {metadata.get('subheading', '')}

            Content:
            {content}
            """
            
            texts.append(enriched_text)
            payloads.append(metadata)

      

        if not texts:
            print("No new documents to index.")
            return

        print("---- FIRST 5 DOCUMENTS BEING INDEXED ----")

        """for i in range(min(5, len(texts))):
            print(f"\n===== SAMPLE {i+1} =====")
            print("CONTENT:\n", texts[i])
            print("METADATA:\n", payloads[i])

        return"""

        # ---------------------------------------------------------
        # Generate Embeddings (Batch Call to Azure)
        # ---------------------------------------------------------
                
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Embedding failed: {e}")
            return

        # ---------------------------------------------------------
        # Prepare Qdrant Points
        # ---------------------------------------------------------
        points = []

        for i, vector in enumerate(vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        **payloads[i],
                        "content": texts[i]   
                    },
                )
            )

        # ---------------------------------------------------------
        # Insert into Qdrant
        # ---------------------------------------------------------
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"{len(points)} documents indexed successfully.")
        except Exception as e:
            print(f"Qdrant upsert failed: {e}")