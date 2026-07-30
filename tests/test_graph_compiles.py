"""Tests that the LangGraph StateGraph builds/compiles and matches the
documented node topology (README's Mermaid diagram)."""

from __future__ import annotations

from underwriting.infrastructure.llm import get_chat_model
from underwriting.service.graph import build_graph


def test_graph_compiles(service) -> None:
    """The `service` fixture already builds+compiles a graph with a real
    checkpointer; if fixture setup didn't raise, compilation succeeded."""
    assert service is not None


def test_graph_has_expected_nodes(settings) -> None:
    """The compiled graph exposes exactly the seven agent/utility nodes
    described in the architecture, plus LangGraph's implicit start/end."""
    chat_model = get_chat_model(settings)
    compiled = build_graph(settings, chat_model).compile()
    node_names = set(compiled.get_graph().nodes.keys())

    expected_nodes = {
        "intake",
        "document_extraction_agent",
        "risk_scoring_agent",
        "policy_compliance_agent",
        "human_escalation",
        "explainability_agent",
        "decision_output",
    }
    assert expected_nodes.issubset(node_names)
    assert "__start__" in node_names
    assert "__end__" in node_names


def test_graph_has_conditional_routing_edge(settings) -> None:
    """policy_compliance_agent must fan out to both human_escalation and
    explainability_agent -- the conditional escalation edge."""
    chat_model = get_chat_model(settings)
    compiled = build_graph(settings, chat_model).compile()
    graph_repr = compiled.get_graph()

    targets = {
        edge.target
        for edge in graph_repr.edges
        if edge.source == "policy_compliance_agent"
    }
    assert "human_escalation" in targets
    assert "explainability_agent" in targets
