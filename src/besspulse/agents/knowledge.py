"""Approved Copilot knowledge tool and tool-vs-RAG routing policy."""

from enum import StrEnum

from rag.schemas import RetrievalResult
from rag.service import KnowledgeService


class QueryRoute(StrEnum):
    OPERATIONAL_TOOL = "OPERATIONAL_TOOL"
    KNOWLEDGE_RAG = "KNOWLEDGE_RAG"
    MIXED = "MIXED"


LIVE_TERMS = {"current", "now", "today", "soc", "latest", "alert", "12h risk", "status"}
KNOWLEDGE_TERMS = {"what is", "what does", "define", "method", "meaning", "limitation", "why"}


def route_query(query: str) -> QueryRoute:
    lowered = query.lower()
    live = any(term in lowered for term in LIVE_TERMS)
    knowledge = any(term in lowered for term in KNOWLEDGE_TERMS)
    if live and knowledge:
        return QueryRoute.MIXED
    return QueryRoute.OPERATIONAL_TOOL if live else QueryRoute.KNOWLEDGE_RAG


def search_knowledge_base(query: str, top_k: int = 4) -> RetrievalResult:
    """Search only the fixed approved-document index; no path argument exists."""

    return KnowledgeService().search_knowledge_base(query, limit=top_k)
