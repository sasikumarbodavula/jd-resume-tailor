"""LangGraph pipeline wiring (spec Section 4.2)."""
from langgraph.graph import END, StateGraph

from app.models.schemas import PipelineState
from app.nodes.ats_score import score_ats_node
from app.nodes.export import export_node
from app.nodes.gap_analysis import gap_analysis_node
from app.nodes.parse_jd import parse_jd_node
from app.nodes.rewrite_content import rewrite_content_node


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("analyze_gap", gap_analysis_node)  # not "gap_analysis": LangGraph forbids node names equal to a state key
    graph.add_node("rewrite_content", rewrite_content_node)
    graph.add_node("score_ats", score_ats_node)
    graph.add_node("export", export_node)

    graph.set_entry_point("parse_jd")
    graph.add_edge("parse_jd", "analyze_gap")
    graph.add_edge("analyze_gap", "rewrite_content")
    graph.add_edge("rewrite_content", "score_ats")
    graph.add_edge("score_ats", "export")  # export only renders when human_approved=True (checked in node)
    graph.add_edge("export", END)
    return graph.compile()


app = build_graph()
