# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------
# Azure OpenAI - Chat Model
# ---------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")


# ---------------------------------------------------
# Azure OpenAI - Embedding Model
# ---------------------------------------------------

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME_EMBEDDING")
AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv("API_VERSION_EMBEDDING")


# ---------------------------------------------------
# Qdrant Configuration
# ---------------------------------------------------

QDRANT_URL = os.getenv("QDARNT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDARNT_API_KEY")

QDRANT_COLLECTION = "hr_policy_chunks"


# ---------------------------------------------------
# LLM Behaviour Config
# ---------------------------------------------------

TEMPERATURE = 0
MAX_TOKENS = 1200


# ---------------------------------------------------
# Retrieval Config
# ---------------------------------------------------

TOP_K = 5
RERANK_TOP_K = 10

# ---------------------------------------------------
# Data Paths
# ---------------------------------------------------

POLICY_DOCS_PATH = "data/policy_docs"