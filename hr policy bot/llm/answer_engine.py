# llm/answer_engine.py

from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)


class AnswerEngine:

    def __init__(self):

        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

    # ---------------------------------------------------------
    # Generate Grounded Answer
    # ---------------------------------------------------------
    def generate(self, query: str, context: str):

        prompt = f"""
You are an HR Policy Assistant.

You must strictly follow these rules:

1. Use ONLY the information provided in the retrieved sections.
2. Do NOT use outside knowledge.
3. If the answer is not found in the retrieved sections, say:
   "The requested information is not available in the policy documents."
4. Ignore irrelevant sections.
5. Provide citations in the format:
   (Document Name - Page Number)

User Query:
{query}

Retrieved Policy Sections:
{context}

Step 1: Carefully analyze the retrieved sections.
Step 2: Identify relevant sections.
Step 3: Extract the answer.
Step 4: Provide a clear and concise answer.
Step 5: Add citations at the end.
"""

        response = self.llm.invoke(prompt)

        answer_text = response.content.strip()

        # Extract sources from context (simple extraction)
        sources = self.extract_sources(context)

        return {
            "answer": answer_text,
            "sources": sources
        }

    # ---------------------------------------------------------
    # Extract Sources from Context
    # ---------------------------------------------------------
    def extract_sources(self, context: str):

        sources = []
        lines = context.split("\n")

        for line in lines:
            if "Document:" in line and "Page:" in line:
                sources.append(line.strip())

        return list(set(sources))