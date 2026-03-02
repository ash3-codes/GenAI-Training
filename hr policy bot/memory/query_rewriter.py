# memory/query_rewriter.py

from llm.azure_config import get_llm


class QueryRewriter:

    def __init__(self):
        self.llm = get_llm()

    def rewrite(self, query: str, conversation_history: str) -> str:

        if not conversation_history.strip():
            return query

        prompt = f"""
You are a query rewriting assistant.

Given conversation history and a new user question,
rewrite the question into a fully standalone question
that can be understood without prior context.

Conversation History:
{conversation_history}

User Question:
{query}

Return only the rewritten standalone question.
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()