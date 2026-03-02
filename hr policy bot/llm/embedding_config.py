# llm/embedding_config.py

from langchain_openai import AzureOpenAIEmbeddings
import config.settings as settings


def get_embeddings():

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_EMBEDDING_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION_EMBEDDING,
    )