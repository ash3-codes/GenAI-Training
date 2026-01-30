import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_version = os.getenv("API_VERSION")
key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("DEPLOYMENT_NAME")

llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    azure_endpoint=endpoint,
    api_key=key,
    api_version=api_version,
    temperature=0,
    max_retries=5,
)
