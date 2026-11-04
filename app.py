"""FastAPI service.  Run:  uvicorn app:app --reload

POST /classify  -> intent, domain, confidence, reason
POST /route     -> route/action + intent + domain
Both accept an optional session_id. A question asked again later in the same session (even after many
other questions, and even if reworded) gets the FIRST answer back unchanged, with "repeated": true,
the earlier "matched_question" and a "similarity" score, and no new LLM call or tokens.
"""
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from classifier import IntentClassifier
from history import SessionHistory, is_storable
from router import Router
from similarity import matcher_from_env

app = FastAPI(title="POC-013 Question Intent Classifier & Router", version="0.3.0")
classifier = IntentClassifier(backend="auto")
router = Router()
matcher = matcher_from_env()      # SIMILARITY_BACKEND=auto|lexical|embedding|off, SIMILARITY_THRESHOLD
history = SessionHistory(per_session=int(os.getenv("HISTORY_MAX", "200")),
                         ttl=float(os.getenv("HISTORY_TTL", "86400")), matcher=matcher)


class QueryIn(BaseModel):
    query: str = Field(..., max_length=2000)
    previous_turn: Optional[str] = Field(None, max_length=4000,
                                         description="Last assistant message, for follow-ups.")
    session_id: Optional[str] = Field(None, max_length=128,
                                      description="Conversation id; repeated questions are remembered per session.")


def _classify(body: QueryIn):
    """Returns (classification, repeat info). A repeat or rephrasing is answered from history, not the LLM."""
    found = history.lookup(body.session_id, body.query)
    if found is not None:
        return found.value, {"repeated": True, "matched_question": found.matched_question,
                             "similarity": found.score}
    c = classifier.classify(body.query, body.previous_turn)
    if is_storable(c, body.previous_turn):
        history.put(body.session_id, body.query, c)
    return c, {"repeated": False, "matched_question": None, "similarity": None}


@app.get("/health")
def health():
    return {"status": "ok", "backend": classifier.backend, "remembered_questions": len(history),
            "similarity": matcher.name if matcher else "off"}


@app.post("/classify")
def classify(body: QueryIn):
    c, rep = _classify(body)
    return {**c.to_dict(), **rep}


@app.post("/route")
def route(body: QueryIn):
    c, rep = _classify(body)
    return {**router.route(body.query, c).to_dict(), **rep}
