from types import SimpleNamespace

from fastapi.testclient import TestClient

import classifier as C
from app import app
from intents import Intent
from router import Router

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_route_each_intent():
    cases = {
        "What is the refund policy?": ("doc_retrieval", "call_rag_pipeline"),
        "How many athletes registered?": ("structured_data", "call_text_to_sql"),
        "Thanks!": ("conversation", "call_chat_llm"),
        "Tell me a joke": ("unsupported", "reply_static"),
        "I want to speak to a human": ("escalation", "handoff_to_human"),
    }
    for q, (route, action) in cases.items():
        r = client.post("/route", json={"query": q}).json()
        assert (r["route"], r["action"]) == (route, action), q


def test_empty_query_does_not_crash():
    assert client.post("/classify", json={"query": "   "}).status_code == 200


def test_low_confidence_clarifies_but_escalation_never_downgraded():
    r = Router(threshold=0.9)
    low_doc = C.Classification(Intent.DOC_RETRIEVAL, 0.5, "x", "llm")
    low_esc = C.Classification(Intent.ESCALATION, 0.5, "x", "llm")
    assert r.route("q", low_doc).route == "clarify"
    assert r.route("q", low_esc).action == "handoff_to_human"


def test_llm_backend_parses_tool_output(monkeypatch):
    clf = C.IntentClassifier.__new__(C.IntentClassifier)
    clf.backend = "llm"
    fake = SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", input={"intent": "STRUCTURED_DATA", "confidence": 0.93, "reason": "count"})])
    llm = C.LLMClassifier.__new__(C.LLMClassifier)
    llm.model, llm.system = "m", "s"
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: fake))
    clf._llm = llm
    out = clf.classify("how many?")
    assert out.intent is Intent.STRUCTURED_DATA and out.backend == "anthropic"


def test_openai_compat_parses_json_mode(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = C.OpenAICompatClassifier("mistral")
    payload = {"choices": [{"message": {"content":
               '{"intent": "escalation", "confidence": 0.97, "reason": "wants human"}'}}]}
    fake = SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)
    llm.http = SimpleNamespace(post=lambda url, json: fake)
    out = llm.classify("get me a person")
    assert out.intent is Intent.ESCALATION and out.backend == "mistral"


def test_llm_failure_falls_back_to_rules():
    clf = C.IntentClassifier.__new__(C.IntentClassifier)
    clf.backend = "llm"

    def boom(*a, **k):
        raise RuntimeError("down")

    clf._llm = SimpleNamespace(classify=boom)
    out = clf.classify("Hello")
    assert out.backend == "rules_fallback"


# ---------------------------------------------------------------- domain + repeated questions
def _fake_app(monkeypatch):
    import app as appmod
    from history import SessionHistory
    calls = []

    class Fake:
        backend = "fake"

        def classify(self, q, prev=None):
            calls.append(q)
            intent = Intent.DOC_RETRIEVAL if len(calls) == 1 else Intent.STRUCTURED_DATA  # differs per call
            return C.Classification(intent, 0.9, "r", "fake", "Finance")

    monkeypatch.setattr(appmod, "classifier", Fake())
    monkeypatch.setattr(appmod, "history", SessionHistory())
    return calls


def test_output_has_domain(monkeypatch):
    _fake_app(monkeypatch)
    assert client.post("/classify", json={"query": "What is the refund policy?"}).json()["domain"] == "Finance"
    assert client.post("/route", json={"query": "How are invoices paid?"}).json()["domain"] == "Finance"


def test_same_question_after_four_others_returns_first_answer(monkeypatch):
    calls = _fake_app(monkeypatch)
    post = lambda q, s="s1": client.post("/route", json={"query": q, "session_id": s}).json()
    first = post("What is the refund policy?")
    for q in ("How many athletes registered?", "Thanks!", "Tell me a joke", "I want a human"):
        post(q)
    again = post("what is the refund policy")
    assert (first["repeated"], again["repeated"]) == (False, True)
    assert (again["intent"], again["action"], again["domain"]) == (first["intent"], first["action"], first["domain"])
    assert len(calls) == 5                                    # no new LLM call for the repeat
    assert post("What is the refund policy?", s="s2")["repeated"] is False


def test_rephrased_repeat_returns_first_answer_without_llm_call(monkeypatch):
    import app as appmod
    from history import SessionHistory
    from similarity import LexicalMatcher, SimilarityMatcher
    calls = _fake_app(monkeypatch)
    monkeypatch.setattr(appmod, "history", SessionHistory(matcher=SimilarityMatcher(LexicalMatcher())))
    post = lambda q: client.post("/route", json={"query": q, "session_id": "s1"}).json()
    first = post("What is the refund policy?")
    for q in ("How many athletes registered?", "Thanks!", "Tell me a joke", "I want a human"):
        post(q)
    again = post("Tell me about the refund rules")
    assert again["repeated"] is True and again["matched_question"] == "What is the refund policy?"
    assert again["similarity"] >= 0.8
    assert (again["intent"], again["domain"]) == (first["intent"], first["domain"])
    assert len(calls) == 5                                     # no new LLM call
    assert post("How many refunds were issued?")["repeated"] is False
