"""Decide whether a new question is a REPHRASING of one already answered in this session.

Runs locally: no LLM call and no API tokens. Two stages:

1. GUARD (rules). Two questions can only match if they are the same KIND of request and mention the same
   specifics. It compares: request-type cues (count/list vs action vs human/complaint vs greeting),
   numbers and IDs, named entities (Dubai, Kona, PO 77412), time words (last/this/next year) and negation.
   This is what stops lookalikes with different answers from matching:
   "What is the refund policy?" vs "How many refunds were issued?"  /  "athlete 48213" vs "athlete 48214".
2. SCORE. LexicalMatcher: word overlap after stop-word removal, stemming and a small IRONMAN synonym table
   (no downloads). EmbeddingMatcher: a small local sentence-embedding model (optional: pip install fastembed)
   which also catches rewordings that share no keywords ("how do I get my money back" ~ "refund policy").

The guard is deliberately strict: a missed paraphrase just costs one normal LLM call, a wrong match returns
the wrong answer.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("poc013.similarity")

# ----------------------------------------------------------------------------- guard
_COUNT = re.compile(
    r"\b(how many|how much|number of|count|total|average|median|percentage|top \d+|rank\w*|list|show|"
    r"which (?:\w+ ){0,2}(?:have|has|are|were|had)|per (?:country|venue|event|shift|hour))\b", re.I)
_ACTION = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+|i (?:want|need|would like) to\s+)*"
    r"(?:buy|book|transfer|cancel|reset|install|send|post|approve|assign|delete|pay|wire|order|start|turn|"
    r"place|email|write|draft|generate|schedule|set up|change|update|remove|add|create|submit|file|refund|diagnose)\b"
    # "Refund policy?" / "Transfer process?" name a topic; they are not commands
    r"(?!\s+(?:polic\w+|rules?|process\w*|procedures?|terms|options?|status|deadlines?|windows?|fees?|limits?|"
    r"forms?|guide\w*|steps?|requirements?|period|dates?|times?|information|info|details?)\b)",
    re.I)
_HUMAN = re.compile(
    r"\b(human|real person|actual person|live agent|manager|supervisor|representative|lawyer|attorney|sue|"
    r"legal action|complain\w*|escalat\w*|emergency|harass\w*|fraud\w*|hacked|scam\w*|unconscious|collapsed|"
    r"on fire|ambulance)\b", re.I)
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|ok|okay|great|bye|goodbye|good (morning|afternoon|evening))\b", re.I)
_TIME = {"last", "this", "next", "previous", "current", "yesterday", "today", "tomorrow", "upcoming", "past",
         "recent", "latest", "earlier"}
_NEG = {"not", "no", "never", "dont", "doesnt", "cant", "cannot", "wont", "isnt", "without", "didnt"}
_NOT_ENTITY = {"i", "i'm", "i'd", "i've", "i'll", "ironman"}


def _words(q: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", q or "")


def signature(q: str) -> dict:
    words = _words(q)
    low = [w.lower().replace("'", "") for w in words]
    cues = frozenset(n for n, rx in (("count", _COUNT), ("action", _ACTION), ("human", _HUMAN),
                                     ("greeting", _GREETING)) if rx.search(q or ""))
    entities = {w.lower() for i, w in enumerate(words)
                if i > 0 and (w[0].isupper() or w.isupper()) and w.lower() not in _NOT_ENTITY}
    return {
        "cues": cues,
        "numbers": frozenset(re.findall(r"\d+(?:[.,]\d+)?", q or "")),
        "entities": entities,
        "first": low[0] if low else "",
        "time": frozenset(w for w in low if w in _TIME),
        "neg": any(w in _NEG for w in low),
    }


def guard_ok(a: str, b: str) -> bool:
    sa, sb = signature(a), signature(b)
    if sa["cues"] != sb["cues"] or sa["numbers"] != sb["numbers"]:
        return False
    if sa["time"] != sb["time"] or sa["neg"] != sb["neg"]:
        return False
    # a sentence-initial capital is not an entity, so ignore the other side's first word when comparing
    return (sa["entities"] - {sb["first"]}) == (sb["entities"] - {sa["first"]})


# ----------------------------------------------------------------------------- lexical score
_STOP = set("""a an the is are was were be been being do does did done i me my mine we our you your it its
this that these those of to for in on at by with from as and or but if so than then there here can could would
will shall may might should must what whats which who whom whose when where why how tell tells about please
kindly explain describe give gives info information details detail regarding need needs want wants know like
also just any some there s ve ll re d m hi hello hey thanks thank am they their them his her theirs""".split())

_PHRASES = [(re.compile(p, re.I), r) for p, r in (
    (r"\bhow many\b", " count "), (r"\bnumber of\b", " count "), (r"\bhow much\b", " count "),
    (r"\bsign[- ]?ups?\b", " register "), (r"\bmoney back\b", " refund "), (r"\bget back my money\b", " refund "),
    (r"\bcut[- ]?off\b", " deadline "), (r"\bwhat'?s\b", " "),
    (r"\b(another|other|some|any) (person|athlete|participant|friend)\b|\b(some|any)(one|body) else\b",
     " otherperson "),
)]
_GROUPS = {
    "policy": "policy policies rule rules guideline guidelines regulation regulations requirement requirements "
              "procedure procedures process processes terms",
    "refund": "refund refunds refunded reimburse reimbursed reimbursement repay repayment",
    "register": "register registered registering registration registrations enrol enroll enrolled enrollment "
                "entry entries",
    "cancel": "cancel cancelled canceled cancellation cancellations cancelling",
    "athlete": "athlete athletes participant participants competitor competitors racer racers triathlete triathletes",
    "vendor": "vendor vendors supplier suppliers",
    "laptop": "laptop laptops computer computers pc notebook",
    "hotel": "hotel hotels accommodation accommodations lodging",
    "flight": "flight flights airfare",
    "count": "count total amount quantity",
    "list": "list lists show display enumerate",
    "start": "start begin commence",
    "cost": "cost costs price prices fee fees charge charges",
    "deadline": "deadline deadlines",
    "leave": "leave vacation holiday pto",
    "pay": "pay paid payment payments",
    "volunteer": "volunteer volunteers",
    "bike": "bike bikes bicycle bicycles cycle cycling",
    "fix": "fix repair resolve",
    "help": "help assist assistance support",
    "speak": "talk speak chat contact call",
}
_CANON = {w: k for k, ws in _GROUPS.items() for w in ws.split()}


def _stem(w: str) -> str:
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def content_tokens(q: str) -> frozenset:
    s = q or ""
    for rx, rep in _PHRASES:
        s = rx.sub(rep, s)
    out = set()
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        if w in _STOP:
            continue
        out.add(_CANON.get(w) or _CANON.get(_stem(w)) or _stem(w))
    return frozenset(out)


class LexicalMatcher:
    name = "lexical"
    default_threshold = 0.80

    def similarity(self, a: str, b: str) -> float:
        ta, tb = content_tokens(a), content_tokens(b)
        if len(ta) < 2 or len(tb) < 2:
            return 0.0                                 # too little content to judge: exact match only
        return len(ta & tb) / len(ta | tb)


class EmbeddingMatcher:
    """Local sentence embeddings (fastembed / ONNX, no torch, no API). First use downloads ~130 MB once."""
    name = "embedding"
    default_threshold = 0.88

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model=None):
        if model is None:
            from fastembed import TextEmbedding           # raises ImportError if not installed

            model = TextEmbedding(model_name)
        self._model = model
        self._cache: dict[str, list[float]] = {}

    def _vec(self, text: str) -> list[float]:
        v = self._cache.get(text)
        if v is None:
            raw = next(iter(self._model.embed([text])))
            v = [float(x) for x in raw]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / n for x in v]
            if len(self._cache) > 5000:
                self._cache.clear()
            self._cache[text] = v
        return v

    def similarity(self, a: str, b: str) -> float:
        va, vb = self._vec(a), self._vec(b)
        return sum(x * y for x, y in zip(va, vb))


# ----------------------------------------------------------------------------- public matcher
@dataclass
class Hit:
    index: int
    score: float


class SimilarityMatcher:
    def __init__(self, scorer, threshold: Optional[float] = None):
        self.scorer = scorer
        self.threshold = scorer.default_threshold if threshold is None else threshold

    @property
    def name(self) -> str:
        return self.scorer.name

    def best(self, question: str, candidates: Sequence[str]) -> Optional[Hit]:
        best: Optional[Hit] = None
        for i, cand in enumerate(candidates):
            if not guard_ok(question, cand):
                continue
            s = self.scorer.similarity(question, cand)
            if s >= self.threshold and (best is None or s > best.score):
                best = Hit(i, s)
        return best


def make_matcher(backend: str = "auto", threshold: Optional[float] = None) -> Optional[SimilarityMatcher]:
    """backend: off | lexical | embedding | auto (embedding if fastembed works, else lexical)."""
    backend = (backend or "auto").lower()
    if backend in ("off", "none", "0", "false"):
        return None
    if backend in ("embedding", "auto"):
        try:
            return SimilarityMatcher(EmbeddingMatcher(), threshold)
        except Exception as e:
            if backend == "embedding":
                raise RuntimeError(f"embedding matcher unavailable ({e}); pip install fastembed") from e
            log.info("fastembed not usable (%s); using the lexical matcher", type(e).__name__)
    return SimilarityMatcher(LexicalMatcher(), threshold)


def matcher_from_env() -> Optional[SimilarityMatcher]:
    t = os.getenv("SIMILARITY_THRESHOLD")
    return make_matcher(os.getenv("SIMILARITY_BACKEND", "auto"), float(t) if t else None)
