# llm/followup_rewriter.py

from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)


class FollowUpRewriter:
    """
    Rewrites short/ambiguous follow-up questions into standalone queries
    using conversation history.
    """

    def __init__(self):

        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

    def rewrite(self, query: str, memory_context: str) -> str:
        """
        Converts follow-up query into standalone query.
        """

        prompt = f"""
You are an HR policy assistant.

Rewrite the user's latest question into a fully standalone question.

Rules:
- Use conversation history if necessary.
- Do NOT answer the question.
- Do NOT summarize.
- Do NOT add extra information.
- Preserve corporate terms exactly.
- Return ONLY the rewritten standalone question.

Conversation History:
{memory_context}

Latest Question:
{query}
"""

        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()

            # Fallback safeguard
            if not rewritten:
                return query

            return rewritten

        except Exception:
            # If LLM fails, return original query
            return query