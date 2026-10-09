"""Grounded retrieval UI with honest operational-tool limitations."""

import streamlit as st

from dashboard.api_client import DashboardAPIError
from dashboard.components.badges import provenance_badges
from dashboard.context import DashboardContext

SUGGESTIONS = (
    "What does technical availability mean?",
    "What are the limitations of the 24h risk model?",
    "How is RevenueAtRisk calculated?",
    "Why do we use chronological splits?",
    "How should I interpret a thermal residual?",
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## AI Copilot")
    provenance_badges("SIMULATED", "REAL", "DERIVED", "MODEL_PREDICTION", "COUNTERFACTUAL")
    st.info(
        "Knowledge retrieval is available when RAG_ENABLED=1. Prompt 13 agent orchestration is not present in this repository."
    )
    st.markdown("### Suggested questions")
    columns = st.columns(3)
    for index, question in enumerate(SUGGESTIONS):
        if columns[index % 3].button(question, key=f"suggest-{index}"):
            st.session_state["copilot_draft"] = question
    for message in st.session_state.get("copilot_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    query = st.chat_input("Ask about battery condition, risk, or market context")
    query = query or st.session_state.pop("copilot_draft", None)
    if query:
        st.session_state["copilot_history"].append({"role": "user", "content": query})
        try:
            response = ctx.client.copilot_query(ctx.asset_id, query)
            message = (
                response.answer or "No approved knowledge-base evidence matched this question."
            )
            st.session_state["copilot_history"].append(
                {
                    "role": "assistant",
                    "content": message,
                    "citations": response.citations,
                    "tool_calls": response.tool_calls,
                }
            )
        except DashboardAPIError as exc:
            if exc.status_code == 503:
                message = (
                    "The required Copilot capability is unavailable. "
                    "No current state or generated answer was inferred."
                )
            else:
                message = f"Copilot API is unavailable ({exc.category}). No generated answer was produced."
            st.session_state["copilot_history"].append({"role": "assistant", "content": message})
            st.rerun()
    with st.expander("Sources / citations"):
        citations = [
            citation
            for item in st.session_state.get("copilot_history", [])
            for citation in item.get("citations", ())
        ]
        if citations:
            for citation in citations:
                st.markdown(f"- {citation}")
        else:
            st.caption("No citations returned yet.")
    with st.expander("Tool calls"):
        calls = [
            call
            for item in st.session_state.get("copilot_history", [])
            for call in item.get("tool_calls", ())
        ]
        if calls:
            for call in calls:
                st.json(call)
        else:
            st.caption("No approved tool calls have been executed.")
