# llm/query_intelligence.py

import re
from typing import Dict
from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)


class QueryIntelligence:

    # ---------------------------------------------------------
    # Acronym Map (Deterministic)
    # ---------------------------------------------------------
    ACRONYM_MAP: Dict[str, str] = {
        "posh": "Prevention of Sexual Harassment (POSH)",
        "fnf": "Full and Final Settlement (FNF)",
        "bgv": "Background Verification (BGV)",
        "lop": "Loss of Pay (LOP)",
        "pip": "Performance Improvement Plan (PIP)",
        "wfh": "Work From Home (WFH)",
        "sla": "Service Level Agreement (SLA)",
        "ctc": "Cost to Company (CTC)",
        "hr": "Human Resources (HR)",
    }

    def __init__(self):

        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=0,
        )

    # ---------------------------------------------------------
    # Basic Cleaning
    # ---------------------------------------------------------
    def clean_query(self, query: str) -> str:
        query = query.strip()
        query = re.sub(r"\s+", " ", query)
        return query

    # ---------------------------------------------------------
    # Deterministic Acronym Expansion
    # ---------------------------------------------------------
    def expand_acronyms(self, query: str) -> str:

        words = query.split()
        expanded_words = []

        for word in words:
            key = re.sub(r"[^\w]", "", word).lower()

            if key in self.ACRONYM_MAP:
                expanded_words.append(self.ACRONYM_MAP[key])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    # ---------------------------------------------------------
    # LLM Spell Correction 
    # ---------------------------------------------------------
    def spell_correct(self, query: str) -> str:

        prompt = f"""
You are an HR assistant.

Your task:
- Correct spelling mistakes.
- Do NOT rewrite or summarize.
- Do NOT change meaning.
- Preserve corporate terms exactly as written.
- Return only the corrected query.

Query:
{query}
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    # ---------------------------------------------------------
    # Intent Classification
    # ---------------------------------------------------------
    def classify_intent(self, query: str) -> str:

        prompt = f"""
Classify the user's query into ONE of the following categories:

- greeting
- identity
- policy_lookup
- summary_request
- unknown

Return ONLY the category name.

Query:
{query}
"""

        response = self.llm.invoke(prompt)
        intent = response.content.strip().lower()

        valid_intents = [
            "greeting",
            "identity",
            "policy_lookup",
            "summary_request",
            "unknown",
        ]

        if intent not in valid_intents:
            return "unknown"

        return intent

    # ---------------------------------------------------------
    # Full Processing Pipeline
    # ---------------------------------------------------------
    def process(self, query: str) -> Dict[str, str]:

        raw_query = query

        cleaned = self.clean_query(raw_query)
        expanded = self.expand_acronyms(cleaned)
        corrected = self.spell_correct(expanded)
        intent = self.classify_intent(corrected)

        return {
            "raw_query": raw_query,
            "normalized_query": corrected,
            "intent": intent,
        }