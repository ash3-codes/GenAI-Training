# llm/azure_config.py

from langchain_openai import AzureChatOpenAI
import config.settings as settings


def get_llm():

    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_CHAT_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )