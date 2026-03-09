"""
graphs/ingestion_graph.py
-------------------------
Ingestion Pipeline — LangGraph compiled graph.

Flow:
  load_documents → parse_resume → validate_schema → expand_skills → embed_and_store

Each node is a LangGraph node. The graph is compiled once at module level
and reused across calls (StateGraph compilation is expensive).

Usage:
    from graphs.ingestion_graph import ingestion_graph

    result = ingestion_graph.invoke({
        "resume_file_paths": ["data/resumes/john.pdf", ...],
        "node_logs":  [],
        "failed_docs": [],
    })
    # result["expanded_resumes"] — stored in Qdrant
    # result["failed_docs"]      — anything that failed along the way
    # result["node_logs"]        — timing + metadata for each node
"""

from langgraph.graph import StateGraph, END

from graphs.state          import PipelineState
from nodes.load_documents  import load_documents_node
from nodes.checkpoint_filter import checkpoint_filter_node
from nodes.parse_resume    import parse_resume_node
from nodes.validate_schema import validate_schema_node
from nodes.expand_skills   import expand_skills_node
from nodes.embed_and_store import embed_and_store_node


def build_ingestion_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("load_documents",    load_documents_node)
    graph.add_node("checkpoint_filter", checkpoint_filter_node)  # NEW: skip already-indexed
    graph.add_node("parse_resume",      parse_resume_node)       # now parallel (ThreadPoolExecutor)
    graph.add_node("validate_schema",   validate_schema_node)
    graph.add_node("expand_skills",     expand_skills_node)
    graph.add_node("embed_and_store",   embed_and_store_node)    # now batch_size=50

    # Pipeline:  load → checkpoint → parse(parallel) → validate → expand → embed(batch=50)
    graph.set_entry_point("load_documents")
    graph.add_edge("load_documents",    "checkpoint_filter")
    graph.add_edge("checkpoint_filter", "parse_resume")
    graph.add_edge("parse_resume",      "validate_schema")
    graph.add_edge("validate_schema",   "expand_skills")
    graph.add_edge("expand_skills",     "embed_and_store")
    graph.add_edge("embed_and_store",   END)

    return graph.compile()


# Compile once at module level — import and reuse
ingestion_graph = build_ingestion_graph()