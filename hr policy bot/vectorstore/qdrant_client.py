# vectorstore/qdrant_client.py

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import config.settings as settings

load_dotenv()



def get_qdrant_client():
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )