"""Router: maps a classification to a downstream handler.

Handlers are stubs for the POC. Swap each for the real pipeline
(RAG search, text-to-SQL, chat LLM, ticketing/handoff) by registering a callable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

from classifier import Classification
from intents import Intent

CONFIDENCE_THRESHOLD = 0.6


@dataclass
class RouteResult:
    route: str            # handler name
    intent: str           # intent that was acted on (may differ from classified one)
    action: str           # what the caller should do next
    response: Optional[str]
    low_confidence: bool
    classification: dict
    domain: str = "General"   # business function the question is about

    def to_dict(self) -> dict:
        return asdict(self)


Handler = Callable[[str, Classification], tuple[str, Optional[str]]]


def _doc_retrieval(q, c):
    return "call_rag_pipeline", None


def _structured(q, c):
    return "call_text_to_sql", None


def _conversation(q, c):
    return "call_chat_llm", None


def _unsupported(q, c):
    return "reply_static", (
        "Sorry, I can't help with that. I can answer questions about event policies and "
        "documents, or look up registration and event data."
    )


def _escalation(q, c):
    return "handoff_to_human", (
        "I'm connecting you with a member of our team now. "
        "If anyone is in danger, please contact local emergency services immediately."
    )


class Router:
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.handlers: dict[Intent, Handler] = {
            Intent.DOC_RETRIEVAL: _doc_retrieval,
            Intent.STRUCTURED_DATA: _structured,
            Intent.CONVERSATION: _conversation,
            Intent.UNSUPPORTED: _unsupported,
            Intent.ESCALATION: _escalation,
        }

    def register(self, intent: Intent, handler: Handler) -> None:
        self.handlers[intent] = handler

    def route(self, query: str, c: Classification) -> RouteResult:
        low = c.confidence < self.threshold

        # Low confidence: ask the user to clarify, except ESCALATION, where a false
        # positive (a human looks at it) is far cheaper than a miss.
        if low and c.intent is not Intent.ESCALATION:
            return RouteResult(
                route="clarify",
                intent=c.intent.value,
                action="ask_clarifying_question",
                response="Could you tell me a bit more about what you need?",
                low_confidence=True,
                classification=c.to_dict(),
                domain=c.domain,
            )

        action, response = self.handlers[c.intent](query, c)
        return RouteResult(
            route=c.intent.value.lower(),
            intent=c.intent.value,
            action=action,
            response=response,
            low_confidence=low,
            classification=c.to_dict(),
            domain=c.domain,
        )
