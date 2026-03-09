"""
graphs/query_graph.py
---------------------
Query Pipeline — LangGraph compiled graph.

Flow:
  parse_jd → embed_jd → hybrid_retrieve → rerank_candidates → score_ats → fuse_scores

Usage:
    from graphs.query_graph import query_graph

    result = query_graph.invoke({
        "jd_raw_text": "We need a Senior Python Engineer...",
        "node_logs":   [],
        "failed_docs": [],
    })
    # result["final_scores"]  — ranked candidates list, ready for UI
    # result["parsed_jd"]     — structured JD (for display)
    # result["node_logs"]     — per-node timing
"""

from langgraph.graph import StateGraph, END

from graphs.state import PipelineState
from nodes.parse_jd           import parse_jd_node
from nodes.embed_jd           import embed_jd_node
from nodes.hybrid_retrieve    import hybrid_retrieve_node
from nodes.rerank_candidates  import rerank_candidates_node
from nodes.score_ats          import score_ats_node
from nodes.fuse_scores        import fuse_scores_node
from nodes.aggregate_results  import aggregate_results_node


def build_query_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("parse_jd",          parse_jd_node)
    graph.add_node("embed_jd",          embed_jd_node)
    graph.add_node("hybrid_retrieve",   hybrid_retrieve_node)
    graph.add_node("rerank_candidates", rerank_candidates_node)
    graph.add_node("score_ats",         score_ats_node)
    graph.add_node("fuse_scores",       fuse_scores_node)
    graph.add_node("aggregate_results", aggregate_results_node)

    graph.set_entry_point("parse_jd")
    graph.add_edge("parse_jd",          "embed_jd")
    graph.add_edge("embed_jd",          "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve",   "rerank_candidates")
    graph.add_edge("rerank_candidates", "score_ats")
    graph.add_edge("score_ats",         "fuse_scores")
    graph.add_edge("fuse_scores",       "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


# Compile once at module level
query_graph = build_query_graph()