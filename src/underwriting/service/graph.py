"""
LangGraph StateGraph assembly for the underwriting pipeline.

Architecture (matches the Mermaid diagram in README.md)
---------------------------------------------------------
    START -> intake -> document_extraction_agent -> risk_scoring_agent
          -> policy_compliance_agent
          -> [conditional: borderline risk OR failed policy -> human_escalation]
          -> human_escalation -> explainability_agent -> decision_output -> END
          -> [conditional: else -> explainability_agent] (skips escalation)

`human_escalation` is the only node that can durably pause the run (via
`langgraph.types.interrupt`); every other edge is a plain sequential
transition. Keeping the graph topology in one small function (rather than
scattering `add_node`/`add_edge` calls across the codebase) makes the
control flow reviewable in one glance -- important for a system whose
entire value proposition is "the decision path is auditable."
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from underwriting.config import Settings
from underwriting.infrastructure.llm import ResilientChatModel
from underwriting.service.nodes import (
    make_decision_output_node,
    make_document_extraction_node,
    make_explainability_node,
    make_human_escalation_node,
    make_intake_node,
    make_policy_compliance_node,
    make_risk_scoring_node,
    route_after_policy,
)
from underwriting.service.state import UnderwritingState


def build_graph(settings: Settings, chat_model: ResilientChatModel) -> StateGraph:
    """Construct (but do not compile) the underwriting StateGraph.

    Returned uncompiled so callers can choose a checkpointer at compile
    time (`compile_graph`) -- useful in tests, which typically want an
    in-memory `:memory:` SqliteSaver rather than the production file path.
    """
    graph = StateGraph(UnderwritingState)

    graph.add_node("intake", make_intake_node())
    graph.add_node("document_extraction_agent", make_document_extraction_node(chat_model))
    graph.add_node("risk_scoring_agent", make_risk_scoring_node(settings))
    graph.add_node("policy_compliance_agent", make_policy_compliance_node())
    graph.add_node("human_escalation", make_human_escalation_node())
    graph.add_node("explainability_agent", make_explainability_node(chat_model))
    graph.add_node("decision_output", make_decision_output_node())

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "document_extraction_agent")
    graph.add_edge("document_extraction_agent", "risk_scoring_agent")
    graph.add_edge("risk_scoring_agent", "policy_compliance_agent")

    graph.add_conditional_edges(
        "policy_compliance_agent",
        route_after_policy,
        {
            "human_escalation": "human_escalation",
            "explainability_agent": "explainability_agent",
        },
    )

    graph.add_edge("human_escalation", "explainability_agent")
    graph.add_edge("explainability_agent", "decision_output")
    graph.add_edge("decision_output", END)

    return graph


def compile_graph(
    settings: Settings,
    chat_model: ResilientChatModel,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Build and compile the graph with the given checkpointer.

    Compiling with a checkpointer is what makes `interrupt()` in
    `human_escalation` a *durable* pause rather than just an in-process
    early return -- see infrastructure/checkpointer.py.
    """
    return build_graph(settings, chat_model).compile(checkpointer=checkpointer)
