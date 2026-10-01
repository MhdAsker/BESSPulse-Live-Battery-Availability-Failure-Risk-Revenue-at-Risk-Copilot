"""Prompt 13-ready Copilot UI shell without fake reasoning."""

import streamlit as st

from dashboard.api_client import DashboardAPIError
from dashboard.components.badges import provenance_badges
from dashboard.context import DashboardContext

SUGGESTIONS = (
    "What changed in the last six hours?",
    "Why is a rack marked critical?",
    "Which issue has the highest commercial impact?",
    "Can the battery meet today's evening peak?",
    "Explain current German market conditions.",
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## AI Copilot")
    provenance_badges("SIMULATED", "REAL", "DERIVED", "MODEL_PREDICTION", "COUNTERFACTUAL")
    st.info(
        "AI Copilot will be enabled in the next phase. This page exercises the Prompt 11 API contract but does not generate answers."
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
            ctx.client.copilot_query(ctx.asset_id, query)
        except DashboardAPIError as exc:
            if exc.status_code == 503:
                message = "AI Copilot will be enabled in the next phase. No generated answer was produced."
            else:
                message = f"Copilot API is unavailable ({exc.category}). No generated answer was produced."
            st.session_state["copilot_history"].append({"role": "assistant", "content": message})
            st.rerun()
    with st.expander("Sources / citations"):
        st.caption(
            "No citations yet. Prompt 13 will ground responses in typed BESSPulse tools and evidence."
        )
    with st.expander("Tool calls · placeholder"):
        st.caption("No tool calls were executed. The agent backend is not configured in Prompt 12.")
