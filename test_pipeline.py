import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

import classifier as C
import dataset
from intents import Intent


def _ironman_like(tmp_path):
    """Workbook shaped like the IRONMAN sheet: different headers, DEV/TEST, expected route, extra sheet."""
    rows = [
        ("I1", "What is the wetsuit rule?", "Doc Retrieval", "DEV", "call_rag_pipeline"),
        ("I2", "How many athletes registered?", "STRUCTURED_DATA", "DEV", "call_text_to_sql"),
        ("I3", "Thanks!", "conversation", "TEST", "call_chat_llm"),
        ("I4", "Tell me a joke", "UNSUPPORTED", "TEST", "reply_static"),
        ("I5", "Get me a human now", "ESCALATION", "TEST", "handoff_to_human"),
    ]
    df = pd.DataFrame(rows, columns=["Query ID", "Question", "Expected Intent", "Set", "Expected Route"])
    p = tmp_path / "ironman.xlsx"
    with pd.ExcelWriter(p) as xw:
        pd.DataFrame({"notes": ["labeling rules"]}).to_excel(xw, sheet_name="rules", index=False)
        df.to_excel(xw, sheet_name="queries", index=False)
    return str(p)


def test_loader_handles_aliases_and_splits(tmp_path):
    df = dataset.load(_ironman_like(tmp_path))
    assert list(df.split) == ["dev", "dev", "test", "test", "test"]
    assert df.label.iloc[0] == "DOC_RETRIEVAL"          # "Doc Retrieval" normalised
    assert df.expected_route.iloc[4] == "handoff_to_human"


def test_loader_rejects_unknown_label(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"query": ["x"], "intent": ["BILLING"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Unknown intent"):
        dataset.load(str(p))


def test_few_shot_uses_dev_rows_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTSET_PATH", _ironman_like(tmp_path))
    monkeypatch.setattr(dataset, "DEFAULT_PATH", _ironman_like(tmp_path))
    shots = C._few_shot()
    assert "wetsuit" in shots and "How many athletes" in shots
    assert "Tell me a joke" not in shots and "human now" not in shots   # TEST rows never leak


def test_schema_rejects_bad_output():
    with pytest.raises(ValidationError):
        C.IntentResult.model_validate({"intent": "BILLING", "confidence": 0.9})
    with pytest.raises(ValidationError):
        C.IntentResult.model_validate({"intent": "ESCALATION", "confidence": 1.7})
    ok = C.IntentResult.model_validate({"intent": "doc retrieval", "confidence": 0.8})
    assert ok.intent is Intent.DOC_RETRIEVAL


def test_invalid_llm_output_falls_back(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = C.OpenAICompatClassifier("mistral")
    payload = {"choices": [{"message": {"content": '{"intent": "BILLING", "confidence": 0.9}'}}]}
    fake = SimpleNamespace(status_code=200, text="", json=lambda: payload)
    llm.http = SimpleNamespace(post=lambda url, json: fake)
    clf = C.IntentClassifier.__new__(C.IntentClassifier)
    clf.backend, clf._llm = "mistral", llm
    out = clf.classify("Hello")
    assert out.backend == "rules_fallback"


def test_code_fenced_json_is_accepted(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = C.OpenAICompatClassifier("mistral")
    body = '```json\n{"intent": "UNSUPPORTED", "confidence": 0.9, "reason": "joke"}\n```'
    fake = SimpleNamespace(status_code=200, text="", json=lambda: {"choices": [{"message": {"content": body}}]})
    llm.http = SimpleNamespace(post=lambda url, json: fake)
    assert llm.classify("joke").intent is Intent.UNSUPPORTED


def test_evaluate_end_to_end_writes_reports(tmp_path):
    data = _ironman_like(tmp_path)
    r = subprocess.run([sys.executable, "evaluate.py", "--backend", "rules", "--split", "test",
                        "--data", data, "--out", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Accuracy:" in r.stdout and "Routing correct:" in r.stdout
    xl = pd.ExcelFile(tmp_path / "out.xlsx")
    assert {"predictions", "per_intent", "confusion", "routing"} <= set(xl.sheet_names)
    assert set(pd.read_excel(xl, sheet_name="predictions")["PASS/FAIL"]) <= {"PASS", "FAIL"}
    assert (tmp_path / "out.md").read_text().startswith("# Error analysis")


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    import time as _t
    sleeps = []
    monkeypatch.setattr(_t, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = C.OpenAICompatClassifier("mistral")
    limited = SimpleNamespace(status_code=429, headers={}, text='quota exceeded. Please retry in 7.5s')
    ok_body = '{"intent": "DOC_RETRIEVAL", "confidence": 0.9, "reason": "policy"}'
    ok = SimpleNamespace(status_code=200, headers={}, text="",
                         json=lambda: {"choices": [{"message": {"content": ok_body}}]})
    seq = [limited, limited, ok]
    llm.http = SimpleNamespace(post=lambda url, json: seq.pop(0))
    out = llm.classify("what is the policy")
    assert out.intent is Intent.DOC_RETRIEVAL and out.backend == "mistral"
    assert sleeps and abs(sleeps[0] - 8.5) < 0.01     # 7.5s hint + 1s margin


# ---------------------------------------------------------------- v2: multi-domain / shared-word set
V2 = Path(__file__).parent / "ironman_testset_v2.csv"


def test_v2_dataset_is_balanced_multidomain():
    df = dataset.load(str(V2))
    assert len(df) == 200 and df["query"].is_unique
    assert set(df["split"]) == {"dev", "test"}
    assert df.groupby("label").size().eq(40).all()
    assert df["domain"].nunique() >= 12
    # every domain has rows in both splits, so per-domain accuracy is measurable on TEST
    assert (df.groupby("domain")["split"].nunique() == 2).all()
    assert (df["ambiguous_term"] != "").sum() >= 30


def test_v2_few_shot_is_dev_only_and_spread_across_domains():
    df = dataset.load(str(V2))
    pairs = dataset.dev_examples(str(V2))
    dev_queries = set(df[df.split == "dev"]["query"])
    assert pairs and all(q in dev_queries for q, _ in pairs)          # no TEST leakage
    domains = {df[df["query"] == q]["domain"].iloc[0] for q, _ in pairs}
    assert len(domains) >= 6                                          # not dominated by one function


def test_prompt_teaches_shared_word_disambiguation():
    from classifier import build_system_prompt
    p = build_system_prompt()
    assert "never by the topic word" in p and "'ticket'" in p and "'transfer'" in p
    for dom in ("finance", "IT", "HR", "legal"):
        assert dom in p


# ---------------------------------------------------------------- v3: domain in the output
def test_domain_normalisation_and_schema():
    from domains import DOMAINS, normalize_domain
    assert normalize_domain("finance") == "Finance" and normalize_domain("Human Resources") == "HR"
    assert normalize_domain("travel") == "Travel and Accommodation"
    assert normalize_domain("cooking") == "General" and normalize_domain(None) == "General"
    assert C.IntentResult.model_validate({"intent": "doc retrieval", "domain": "it"}).domain == "IT"
    assert C.IntentResult.model_validate({"intent": "DOC_RETRIEVAL"}).domain == "General"   # model omitted it
    assert len(DOMAINS) == len(set(DOMAINS)) == 13


def test_rules_fallback_returns_a_domain():
    assert C.classify_rules("What is the expense reimbursement policy?").domain == "Finance"
    assert C.classify_rules("How do I set up the VPN on my laptop?").domain == "IT"
    assert C.classify_rules("Thanks!").domain == "General"
    assert C.classify_rules("Tell me a joke").domain == "General"


def test_openai_compat_reads_domain(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    llm = C.OpenAICompatClassifier("mistral")
    payload = {"choices": [{"message": {"content":
               '{"intent": "doc_retrieval", "domain": "HR", "confidence": 0.9, "reason": "leave rule"}'}}]}
    fake = SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)
    llm.http = SimpleNamespace(post=lambda url, json: fake)
    out = llm.classify("What is the leave policy?")
    assert out.domain == "HR" and out.to_dict()["domain"] == "HR"
    assert '"domain"' in llm.system                      # the JSON instruction asks for it


def test_router_output_carries_domain():
    from router import Router
    c = C.Classification(Intent.STRUCTURED_DATA, 0.9, "x", "fake", "Finance")
    assert Router().route("q", c).to_dict()["domain"] == "Finance"
    low = C.Classification(Intent.DOC_RETRIEVAL, 0.3, "x", "fake", "IT")
    assert Router().route("q", low).domain == "IT"       # also on the clarify route


def test_prompt_lists_domains_and_v2_labels_use_them():
    from classifier import build_system_prompt
    from domains import DOMAINS
    p = build_system_prompt()
    assert all(d in p for d in DOMAINS)
    df = dataset.load(str(V2))
    assert set(df["expected_domain"]) <= set(DOMAINS)
    assert dataset.dev_domains(str(V2))


# ---------------------------------------------------------------- v3: same question later -> same answer
def test_history_returns_first_answer_after_other_questions():
    from history import SessionHistory
    h = SessionHistory()
    h.put("s1", "What is the refund policy?", "FIRST")
    for i in range(4):
        h.put("s1", f"another question {i}", f"A{i}")
    assert h.get("s1", "what is the REFUND policy") == "FIRST"        # case and end punctuation ignored
    assert h.get("s2", "What is the refund policy?") is None           # other session: nothing remembered
    h.put("s1", "What is the refund policy?", "SECOND")
    assert h.get("s1", "What is the refund policy?") == "FIRST"        # first answer wins


def test_history_shared_bucket_expiry_and_limits(monkeypatch):
    import history as H
    h = H.SessionHistory(per_session=3, ttl=10)
    for i in range(5):
        h.put(None, f"q{i}", i)
    assert h.get(None, "q0") is None and h.get(None, "q4") == 4         # oldest dropped beyond the limit
    t = [H.time.monotonic()]
    monkeypatch.setattr(H.time, "monotonic", lambda: t[0] + 11)
    assert h.get(None, "q4") is None                                    # expired


def test_history_does_not_store_fallbacks_or_contextual_followups():
    from history import is_storable
    ok = C.Classification(Intent.DOC_RETRIEVAL, 0.9, "x", "gemini")
    assert is_storable(ok, None)
    assert not is_storable(C.Classification(Intent.DOC_RETRIEVAL, 0.5, "x", "rules_fallback"), None)
    followup = C.Classification(Intent.CONVERSATION, 0.9, "x", "gemini")
    assert not is_storable(followup, "previous assistant message")
    assert is_storable(followup, None)                                  # a plain "hi" is fine


# ---------------------------------------------------------------- v4: rephrased repeats (local, no tokens)
def _pairs():
    return pd.read_csv(Path(__file__).parent / "paraphrase_pairs.csv")


def test_lexical_matcher_catches_keyword_rewordings_and_never_matches_lookalikes():
    from similarity import LexicalMatcher, SimilarityMatcher
    m = SimilarityMatcher(LexicalMatcher())
    df = _pairs()
    wrong = [(r.original, r.other) for r in df[df.should_match == 0].itertuples()
             if m.best(r.other, [r.original])]
    assert wrong == []                                        # a wrong match returns a wrong answer
    easy = df[(df.should_match == 1) & (df.tier == "easy")]
    hit = sum(1 for r in easy.itertuples() if m.best(r.other, [r.original]))
    assert hit / len(easy) >= 0.9


@pytest.mark.parametrize("a,b", [
    ("What is the refund policy?", "How many refunds were issued?"),         # rule vs count
    ("What is the transfer policy?", "Transfer my entry now."),              # question vs action
    ("What is the status of athlete ID 48213?", "What is the status of athlete ID 48214?"),
    ("What is the cut-off time for the Dubai race?", "What is the cut-off time for the Kona race?"),
    ("How many athletes registered last year?", "How many athletes registered this year?"),
    ("Which items are allowed in transition?", "Which items are not allowed in transition?"),
])
def test_guard_blocks_lookalikes(a, b):
    from similarity import guard_ok
    assert not guard_ok(a, b)


def test_history_returns_first_answer_for_a_rephrased_repeat_after_other_questions():
    from history import SessionHistory
    from similarity import LexicalMatcher, SimilarityMatcher
    h = SessionHistory(matcher=SimilarityMatcher(LexicalMatcher()))
    h.put("s", "What is the refund policy?", "FIRST")
    for i, q in enumerate(["How many athletes registered?", "Thanks!", "Tell me a joke", "I want a human"]):
        h.put("s", q, f"A{i}")
    m = h.lookup("s", "Tell me about the refund rules")
    assert m.value == "FIRST" and m.kind == "similar" and m.matched_question == "What is the refund policy?"
    assert h.lookup("s", "How many refunds were issued?") is None            # lookalike, different answer
    assert h.lookup("other-session", "Refund policy?") is None
    assert h.lookup("s", "what is the refund policy").kind == "exact"


def test_embedding_matcher_catches_rewordings_without_shared_words_and_guard_still_applies():
    from similarity import EmbeddingMatcher, SimilarityMatcher

    class FakeModel:                       # stands in for the local ONNX model: same vector = same meaning
        VEC = {"What is the refund policy?": [1.0, 0.0], "How do I get my money back?": [0.98, 0.2],
               "How many refunds were issued?": [0.97, 0.24]}

        def embed(self, texts):
            return iter([self.VEC[t] for t in texts])

    m = SimilarityMatcher(EmbeddingMatcher(model=FakeModel()), threshold=0.95)
    assert m.best("How do I get my money back?", ["What is the refund policy?"]) is not None
    # vector is almost identical, but a count question is a different request: the guard refuses
    assert m.best("How many refunds were issued?", ["What is the refund policy?"]) is None


def test_make_matcher_modes(monkeypatch):
    import similarity as sim

    def no_model(*a, **k):
        raise ImportError("fastembed not installed")

    monkeypatch.setattr(sim, "EmbeddingMatcher", no_model)
    assert sim.make_matcher("off") is None
    assert sim.make_matcher("lexical").name == "lexical"
    assert sim.make_matcher("auto").name == "lexical"          # no embedding model: quietly uses lexical
    with pytest.raises(RuntimeError):
        sim.make_matcher("embedding")                           # explicitly requested but unavailable


# ---------------------------------------------------------------- v5: no split/difficulty columns in the sheets
def test_v2_visible_sheet_has_no_split_or_difficulty_but_the_code_still_uses_them():
    here = Path(__file__).parent
    for raw in (pd.read_csv(here / "ironman_testset_v2.csv"), pd.read_excel(here / "ironman_testset_v2.xlsx")):
        assert not {"split", "difficulty"} & set(raw.columns)
    assert "_splits" in pd.ExcelFile(here / "ironman_testset_v2.xlsx").sheet_names
    for f in ("ironman_testset_v2.csv", "ironman_testset_v2.xlsx"):
        df = dataset.load(str(here / f))
        assert df.split.value_counts().to_dict() == {"test": 130, "dev": 70}
        assert (df.difficulty == "hard").sum() == 57
        assert dataset.dev_examples(str(here / f))                      # few-shot still comes from DEV


def test_results_workbook_has_no_split_or_difficulty(tmp_path):
    here = Path(__file__).parent
    out = tmp_path / "res"
    r = subprocess.run([sys.executable, str(here / "evaluate.py"), "--backend", "rules", "--split", "dev",
                        "--data", str(here / "ironman_testset_v2.xlsx"), "--out", str(out)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-400:]
    cols = set(pd.read_excel(f"{out}.xlsx", sheet_name="predictions").columns)
    assert not {"split", "difficulty"} & cols
    assert {"query", "label", "pred", "pred_domain", "PASS/FAIL"} <= cols
    assert len(pd.read_excel(f"{out}.xlsx", sheet_name="predictions")) == 70   # --split dev still selected DEV


def test_hide_columns_tool_keeps_splits_in_a_hidden_sheet(tmp_path):
    here = Path(__file__).parent
    src = tmp_path / "mine.xlsx"
    pd.read_excel(here / "testset.xlsx", sheet_name="testset").to_excel(src, sheet_name="testset", index=False)
    before = dataset.load(str(src))
    r = subprocess.run([sys.executable, str(here / "hide_columns.py"), str(src)], capture_output=True, text=True,
                       cwd=tmp_path, env={**__import__("os").environ, "PYTHONPATH": str(here)})
    assert r.returncode == 0, r.stderr[-400:]
    clean = tmp_path / "mine_clean.xlsx"
    assert not {"split", "difficulty"} & set(pd.read_excel(clean).columns)
    after = dataset.load(str(clean))
    assert list(after["split"]) == list(before["split"]) and list(after["label"]) == list(before["label"])
