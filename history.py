"""Remembers the answer given to each question, so the same question later gets the SAME answer.

Why: the LLM can return a slightly different answer each time, and the user may ask the same thing again
after three or four other questions, possibly worded differently. The first answer is stored per session
(or in one shared bucket when no session_id is sent) and returned unchanged, with no LLM call and no tokens.

Lookup order: 1) same question after cleanup (case, spacing, end punctuation); 2) a paraphrase, decided
locally by similarity.py (strict guard + similarity score). Both are free.

Not stored: fallback answers (a temporary LLM failure must not become permanent) and context-dependent
follow-ups ("explain that simpler"), whose meaning depends on the previous message.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_BUCKET = "_shared"


def normalize_question(q: str) -> str:
    """Case, spacing and end punctuation do not make a different question."""
    s = re.sub(r"\s+", " ", (q or "").strip().lower())
    return s.rstrip(" ?.!,;:")


@dataclass
class Match:
    value: Any
    kind: str                 # "exact" | "similar"
    score: float              # 1.0 for exact
    matched_question: str     # the earlier question whose answer is being returned


class SessionHistory:
    def __init__(self, per_session: int = 200, ttl: float = 86400.0, max_sessions: int = 10000, matcher=None):
        self.per_session, self.ttl, self.max_sessions = per_session, ttl, max_sessions
        self.matcher = matcher                       # similarity.SimilarityMatcher or None (exact only)
        # session -> normalized question -> (stored_at, value, original question text)
        self._data: "OrderedDict[str, OrderedDict[str, tuple[float, Any, str]]]" = OrderedDict()
        self._lock = threading.Lock()

    def lookup(self, session_id: Optional[str], question: str) -> Optional[Match]:
        sid, key = session_id or DEFAULT_BUCKET, normalize_question(question)
        if not key:
            return None
        now = time.monotonic()
        with self._lock:
            bucket = self._data.get(sid)
            if not bucket:
                return None
            for k in [k for k, (stamp, _, _) in bucket.items() if now - stamp > self.ttl]:
                del bucket[k]                        # drop expired entries
            if key in bucket:
                _, value, original = bucket[key]
                self._data.move_to_end(sid)
                return Match(value, "exact", 1.0, original)
            if self.matcher is None or not bucket:
                return None
            snapshot = [(v, orig) for _, v, orig in bucket.values()]   # score outside the lock
        hit = self.matcher.best(question, [orig for _, orig in snapshot])
        if hit is None:
            return None
        value, original = snapshot[hit.index]
        return Match(value, "similar", round(hit.score, 4), original)

    def get(self, session_id: Optional[str], question: str) -> Optional[Any]:
        m = self.lookup(session_id, question)
        return m.value if m else None

    def put(self, session_id: Optional[str], question: str, value: Any) -> None:
        sid, key = session_id or DEFAULT_BUCKET, normalize_question(question)
        if not key:
            return
        with self._lock:
            bucket = self._data.setdefault(sid, OrderedDict())
            if key not in bucket:                    # first answer wins: never overwritten
                bucket[key] = (time.monotonic(), value, re.sub(r"\s+", " ", question.strip()))
                while len(bucket) > self.per_session:
                    bucket.popitem(last=False)
            self._data.move_to_end(sid)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._data.values())


def is_storable(classification, previous_turn: Optional[str]) -> bool:
    """False for fallback answers and for follow-ups that only make sense with the previous message."""
    if str(getattr(classification, "backend", "")).endswith("fallback"):
        return False
    if previous_turn and getattr(classification.intent, "value", "") == "CONVERSATION":
        return False
    return True
