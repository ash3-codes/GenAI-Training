import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_version = os.getenv("API_VERSION")
key = os.getenv("AZURE_OPENAI_API_KEY")
deploment_name = os.getenv("DEPLOYMENT_NAME")

llm = AzureChatOpenAI(
    azure_deployment=deploment_name,
    azure_endpoint=endpoint,
    api_key=key,
    api_version=api_version,
    temperature=0,
    max_retries=5,
)

"""response=llm.invoke('python code to sort a list')
response.usage_metadata     #gives tokens used
print(response.content)

"""


"""#streaming gives better user experience

for chunk in llm.stream("python code to sort a list"):
    print(chunk.text, end="", flush=True)"""

#Streaming and chunks
#During streaming, you’ll receive AIMessageChunk objects that can be combined into a full message object:


chunks = []
full_message = None
for chunk in llm.stream("python code to sort a list"):
    chunks.append(chunk)
    print(chunk.text, end="")
    full_message = chunk if full_message is None else full_message + chunk
    #full_message.usage_metadata  : gives token used

"""response = client.chat.completions.create(
    model="gpt-4.1-mini", # The name of your model deployment
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Azure OpenAI?"}
    ]
)

print(response.choices[0].message.content)
"""

"""from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(model="gpt-4.1-mini", stream_usage=True)"""


"""from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)"""