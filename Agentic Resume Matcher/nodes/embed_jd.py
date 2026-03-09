"""
nodes/embed_jd.py
-----------------
JD Embedding Node — LangGraph node.

Embeds the parsed JD and optionally stores it in jd_index collection.
The JD vector is kept in state (jd_embedding) for immediate use by
the retrieval node — no round-trip to Qdrant needed for the query vector.
"""

import warnings
from utils.logger import NodeTimer


def embed_jd_node(state: dict) -> dict:
    """
    LangGraph node: Embed the parsed JD.

    Reads:  state["parsed_jd"]    — JobDescriptionSchema dict
    Writes: state["jd_embedding"] — list[float] vector (1536 dims)
    """
    parsed_jd: dict | None = state.get("parsed_jd")

    with NodeTimer("embed_jd_node", state) as timer:
        if not parsed_jd:
            timer.extra = {"error": "no_parsed_jd"}
            return {"jd_embedding": None}

        from schemas.jd_schema import JobDescriptionSchema
        from config.settings import get_embedding_model

        # Build embedding text from JD schema
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)
            clean_jd = {k: v for k, v in parsed_jd.items() if k != "parsed"}
            jd_obj = JobDescriptionSchema(**clean_jd)

        embed_text = jd_obj.to_embedding_text()
        embedding_model = get_embedding_model()
        vector = embedding_model.embed_query(embed_text)

        timer.extra = {
            "jd_title":    jd_obj.title,
            "vector_dims": len(vector),
            "embed_text_preview": embed_text[:80],
        }

    return {"jd_embedding": vector}