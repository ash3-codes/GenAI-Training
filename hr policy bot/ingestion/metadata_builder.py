# ingestion/metadata_builder.py

import hashlib
from datetime import datetime
from typing import Dict, Any


class MetadataBuilder:
    """
    Enriches structured + chunked policy blocks with enterprise-ready metadata.
    Designed for secure HR RAG systems.
    """

    def __init__(self, default_access: str = "internal"):
        self.default_access = default_access

    # ------------------------------------------------------------------
    # Generate SHA256 hash for duplicate detection & integrity tracking
    # ------------------------------------------------------------------
    def generate_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Detect content type for filtering & retrieval optimization
    # ------------------------------------------------------------------
    def detect_content_type(self, block: Dict[str, Any]) -> str:
        if block.get("is_table", False):
            return "table"
        if block.get("is_bullet", False):
            return "bullet"
        return "text"

    # ------------------------------------------------------------------
    # Build enriched metadata object
    # ------------------------------------------------------------------
    def enrich(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts a chunked structural block and returns:
        {
            "content": str,
            "metadata": {...}
        }
        """

        content = block.get("content", "").strip()

        metadata = {
            # -------------------------------
            # Document-level information
            # -------------------------------
            "doc_name": block.get("doc_name"),
            "doc_id": block.get("doc_id", block.get("doc_name")),
            "version": block.get("version", "v1.0"),
            "effective_date": block.get("effective_date"),

            # -------------------------------
            # Structural hierarchy
            # -------------------------------
            "page_number": block.get("page_number"),
            "chapter": block.get("chapter"),
            "heading": block.get("heading"),
            "subheading": block.get("subheading"),
            "section_id": block.get("section_id"),

            # -------------------------------
            # Content classification
            # -------------------------------
            "content_type": self.detect_content_type(block),
            "is_table": block.get("is_table", False),
            "is_bullet": block.get("is_bullet", False),

            # -------------------------------
            # Security & access control
            # -------------------------------
            "access_level": block.get("access_level", self.default_access),
            "department_scope": block.get("department_scope", ["All"]),

            # -------------------------------
            # Integrity & auditing
            # -------------------------------
            "content_hash": self.generate_hash(content),
            "ingested_at": datetime.utcnow().isoformat(),
        }

        return {
            "content": content,
            "metadata": metadata
        }

    # ------------------------------------------------------------------
    # Bulk enrichment helper
    # ------------------------------------------------------------------
    def enrich_all(self, blocks: list) -> list:
        """
        Process a list of chunked blocks and return enriched documents.
        """
        enriched_docs = []
        for block in blocks:
            
            content = block.get("content", "")

            if self.is_low_value_chunk(content):
                continue


            enriched_docs.append(self.enrich(block))
        return enriched_docs
    
    # -------------------------------------------------------------------
    # remove noise 
    # -------------------------------------------------------------------

    def is_low_value_chunk(self, content: str) -> bool:

        if not content:
            return True

        text = content.lower().strip()

        # Remove extremely short chunks
        if len(text) < 40:
            return True

        # Table of contents detection
        if "table of contents" in text:
            return True

        if text.startswith("contents"):
            return True

        # Legal disclaimer detection
        if "do not disclose" in text:
            return True

        if "proprietary to" in text:
            return True

        # Version / cover page noise
        if "version" in text and len(text) < 200:
            return True

        # Pure numbering lines like "1.0 Objective"
        if text.replace(".", "").replace(" ", "").isdigit():
            return True

        return False