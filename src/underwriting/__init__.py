"""
Credit Underwriting Decision Graph.

A LangGraph multi-agent system that adjudicates synthetic loan applications
through a pipeline of specialized agents (intake, document extraction, risk
scoring, policy compliance, human escalation, explainability, and decision
output), with durable checkpointing so human-in-the-loop review pauses can
survive process restarts.

This is a portfolio / demonstration project built around a fictional bank,
"Northbridge Financial Group", using entirely synthetic data. It is not
affiliated with, and does not represent the practices of, any real financial
institution.
"""

__version__ = "0.1.0"
