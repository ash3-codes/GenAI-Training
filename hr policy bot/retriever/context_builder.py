# retriever/context_builder.py

from typing import List, Dict
import config.settings as settings


class ContextBuilder:
    """
    Builds structured LLM-ready context
    with citation-friendly formatting.
    """

    def __init__(self, max_context_chars: int = 15000):
        self.max_context_chars = max_context_chars

    def build(self, documents: List[Dict]) -> str:

        if not documents:
            return ""

        context_parts = []
        total_length = 0

        for idx, doc in enumerate(documents):

            content = (doc.get("content") or "").strip()
            metadata = doc.get("metadata", {})

            doc_name = metadata.get("doc_name", "Unknown Document")
            page_number = metadata.get("page_number", "N/A")
            heading = metadata.get("heading", "")
            subheading = metadata.get("subheading", "")

            citation_header = f"""
==============================
Source ID: {idx}
Document: {doc_name}
Page: {page_number}
Section: {heading} {subheading}
==============================
"""

            formatted_chunk = f"""
{citation_header}
{content}
"""

            total_length += len(formatted_chunk)

            if total_length > self.max_context_chars:
                break

            context_parts.append(formatted_chunk)

        final_context = "\n".join(context_parts)

        return final_context