# retriever/gpt_reranker.py

from typing import List, Dict
import json
import re

from llm.azure_config import get_llm
import config.settings as settings


class AzureGPTReranker:
    """
    GPT-based reranker for HR policy chunks.
    Replaces cross-encoder.
    """

    def __init__(self):
        self.llm = get_llm()

    # ---------------------------------------------------------
    # Build strict ranking prompt
    # ---------------------------------------------------------
    def _build_prompt(self, query: str, documents: List[Dict]) -> str:

        formatted_docs = ""

        for idx, doc in enumerate(documents):
            formatted_docs += f"""
Document ID: {idx}
Content:
{doc["content"]}

-------------------------
"""

        prompt = f"""
You are a document relevance ranking system for an internal HR Policy Assistant.

Your task:
Given a user query and multiple HR policy excerpts,
rank the documents from most relevant to least relevant.

IMPORTANT RULES:
- Return ONLY a JSON list of document IDs.
- Do NOT explain.
- Do NOT add commentary.
- Do NOT add text outside JSON.
- Return example format: [3,1,0,2]

User Query:
{query}

Documents:
{formatted_docs}
"""

        return prompt

    # ---------------------------------------------------------
    # Extract JSON safely
    # ---------------------------------------------------------
    def _safe_json_parse(self, text: str):

        try:
            return json.loads(text)
        except Exception:
            # Try extracting JSON array manually
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    return None
            return None

    # ---------------------------------------------------------
    # Rerank documents
    # ---------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:

        if not documents:
            return []

        if top_k is None:
            top_k = settings.TOP_K

        prompt = self._build_prompt(query, documents)

        try:
            response = self.llm.invoke(prompt)
            ranked_indices = self._safe_json_parse(response.content)
        except Exception:
            return documents[:top_k]

        if not ranked_indices or not isinstance(ranked_indices, list):
            return documents[:top_k]

        ranked_docs = []

        for idx in ranked_indices:
            if isinstance(idx, int) and 0 <= idx < len(documents):
                ranked_docs.append(documents[idx])

        # Fallback safety
        if not ranked_docs:
            return documents[:top_k]

        return ranked_docs[:top_k]