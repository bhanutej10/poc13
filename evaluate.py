"""Evaluate the classifier + router on a labeled test set (DEV for tuning, TEST once for the 90% check).

  python evaluate.py --data ironman_testset.xlsx --backend gemini --split dev
  python evaluate.py --data ironman_testset.xlsx --backend gemini --split test     # run ONCE
  python evaluate.py --backend rules --split test                                  # offline smoke test

Outputs (prefix results_<backend>_<split>):
  .xlsx  predictions with PASS/FAIL, per-intent metrics, confusion matrix, routing summary
  .md    error analysis: every misclassification, severity, likely cause, suggested fix
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

import dataset
from classifier import IntentClassifier
from intents import Intent
from router import Router

TARGET = 0.90
LABELS = [i.value for i in Intent]

# Expected-route aliases: a sheet may say "doc_retrieval", "call_rag_pipeline", "RAG", etc.
ROUTE_OF = {
    "DOC_RETRIEVAL": {"doc_retrieval", "call_rag_pipeline", "rag", "doc"},
    "STRUCTURED_DATA": {"structured_data", "call_text_to_sql", "sql", "text_to_sql", "structured"},
    "CONVERSATION": {"conversation", "call_chat_llm", "chat"},
    "UNSUPPORTED": {"unsupported", "reply_static", "static", "reject", "decline"},
    "ESCALATION": {"escalation", "handoff_to_human", "human", "handoff", "escalate"},
}

# Why a pair is commonly confused + what to try. Hints, not diagnoses: verify against the query.
PAIR_HINTS = {
    ("DOC_RETRIEVAL", "STRUCTURED_DATA"): (
        "Policy/rule question worded like a data lookup (numbers, 'how many', 'total').",
        "Add rule + examples: asking what a policy SAYS is DOC_RETRIEVAL; counting/listing records is STRUCTURED_DATA."),
    ("STRUCTURED_DATA", "DOC_RETRIEVAL"): (
        "Data question that mentions a policy/guide word (refund, rules, entry).",
        "Add lookalike pairs: 'refund policy' vs 'total refunds issued'."),
    ("DOC_RETRIEVAL", "UNSUPPORTED"): (
        "In-domain document question read as general knowledge.",
        "Tell the model the domain explicitly and list in-scope topics (race rules, logistics, travel)."),
    ("UNSUPPORTED", "DOC_RETRIEVAL"): (
        "Out-of-scope question that sounds like a how-to/policy question.",
        "Add out-of-scope examples with how-to phrasing; define scope boundaries."),
    ("ESCALATION", "DOC_RETRIEVAL"): (
        "Complaint or emergency phrased as an information question.",
        "Rule: emergencies, complaints, legal threats and 'get me a person' override topic."),
    ("ESCALATION", "STRUCTURED_DATA"): (
        "Escalation mentions an order/registration, so it looks like a record lookup.",
        "Rule: distress or request for a human wins over any topic words."),
    ("ESCALATION", "UNSUPPORTED"): (
        "Complaint about something out of scope.",
        "Rule: any request for a human or formal complaint is ESCALATION first."),
    ("ESCALATION", "CONVERSATION"): (
        "Frustration expressed in a short, chatty message.",
        "Add examples of short angry messages -> ESCALATION."),
    ("CONVERSATION", "DOC_RETRIEVAL"): (
        "Follow-up like 'explain that simpler' treated as a new question.",
        "Pass previous_turn to the classifier; add follow-up examples."),
    ("CONVERSATION", "UNSUPPORTED"): (
        "Small talk about the assistant treated as off-topic.",
        "Define small talk and questions about the assistant as CONVERSATION."),
}


def severity(true: str, pred: str) -> str:
    if true == "ESCALATION":
        return "HIGH - a person needing help was not handed to a human"
    if pred == "ESCALATION":
        return "LOW - unneeded handoff (costs staff time, safe for the user)"
    if true == "UNSUPPORTED":
        return "MEDIUM - system tries to answer something it should decline"
    if pred == "UNSUPPORTED":
        return "MEDIUM - valid question refused"
    return "LOW - wrong pipeline, user gets a poor answer"


def route_ok(true_label, expected_route, result) -> bool:
    """Routing is correct when the router acted on the right handler.
    If the sheet has an expected_route, compare against it; else derive from the true intent."""
    got = {result.route.lower(), result.action.lower()}
    if result.route == "clarify":
        return False
    want = ROUTE_OF[true_label]
    if expected_route and str(expected_route).strip().lower() not in ("", "nan", "none"):
        want = want | {str(expected_route).strip().lower()}
    return bool(got & want)


def write_report(path, df, backend, split, acc, per, cm, rep_pairs):
    wrong = df[~df.correct]
    lines = [
        f"# Error analysis: {backend} on {split.upper()}",
        "",
        f"- Rows: {len(df)} | Correct: {int(df.correct.sum())} | Misclassified: {len(wrong)}",
        f"- Accuracy: **{acc:.1%}** (target {TARGET:.0%}) -> **{'PASS' if acc >= TARGET else 'FAIL'}**",
        f"- Routing correct: **{df.routing_ok.mean():.1%}**",
        "",
        "## Per-intent metrics",
        "",
        per.round(3).to_markdown(index=False),
        "",
        "## Most common confusions (true -> predicted)",
        "",
    ]
    if rep_pairs:
        lines += [f"- {t} -> {p}: {n}" for (t, p), n in rep_pairs]
    else:
        lines.append("- none")
    lines += ["", "## Misclassified queries", ""]
    if wrong.empty:
        lines.append("None.")
    for _, r in wrong.iterrows():
        cause, fix = PAIR_HINTS.get((r.label, r.pred), (
            "No standard pattern; read the query and the model's reason.",
            "Add this query (or a close paraphrase) to DEV few-shot examples and re-run DEV."))
        lines += [
            f"### {r['id']}: {r['query']}",
            f"- Domain: {r.get('domain', '') or 'n/a'} | shared word: {r.get('ambiguous_term', '') or 'none'}",
            f"- Expected: `{r.label}` | Predicted: `{r.pred}` (confidence {r.confidence:.2f}, backend {r.backend_used})",
            f"- Model's reason: {r.reason or 'n/a'}",
            f"- Severity: {severity(r.label, r.pred)}",
            f"- Likely cause (hint): {cause}",
            f"- Suggested fix: {fix}",
            "",
        ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="rules",
                    choices=["rules", "gemini", "mistral", "grok", "anthropic", "auto"])
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--data", default=dataset.DEFAULT_PATH)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", default=None, help="output prefix")
    ap.add_argument("--force-test", action="store_true", help="allow re-running TEST with an LLM backend")
    a = ap.parse_args()

    try:
        df = dataset.load(a.data, a.sheet)
    except (FileNotFoundError, ValueError) as ex:
        sys.exit(f"ERROR: {ex}")
    if a.split != "all":
        df = df[df.split == a.split].copy()
    if df.empty:
        sys.exit(f"No rows for split={a.split}. Splits present: {sorted(dataset.load(a.data, a.sheet).split.unique())}")

    # TEST is the held-out 90% check: run it once per LLM backend, otherwise you are tuning on it.
    lock = Path(f".test_run_{a.backend}.lock")
    if a.split in ("test", "all") and a.backend != "rules":
        if lock.exists() and not a.force_test:
            sys.exit(f"TEST already run with {a.backend} ({lock.read_text().strip()}). "
                     f"Re-running means tuning on TEST. Use --force-test only if you accept that.")

    clf = IntentClassifier(backend=a.backend)
    router = Router()
    preds, results = [], []
    for q in df["query"]:
        c = clf.classify(q)
        preds.append(c)
        results.append(router.route(q, c))

    df["pred"] = [p.intent.value for p in preds]
    df["confidence"] = [p.confidence for p in preds]
    df["reason"] = [p.reason for p in preds]
    df["backend_used"] = [p.backend for p in preds]
    df["pred_domain"] = [p.domain for p in preds]
    df["routed_to"] = [r.route for r in results]
    df["action"] = [r.action for r in results]
    df["correct"] = df.pred == df.label
    df["routing_ok"] = [route_ok(l, e, r) for l, e, r in zip(df.label, df.expected_route, results)]
    df["PASS/FAIL"] = df.correct.map({True: "PASS", False: "FAIL"})

    fb = int((df.backend_used == "rules_fallback").sum())
    if a.backend != "rules" and fb == len(df):
        sys.exit(f"\nERROR: every LLM call failed, so there is no LLM accuracy to report.\n"
                 f"First error: {getattr(clf._llm, 'last_error', None) or getattr(clf, 'last_error', 'unknown')}")
    if fb:
        print(f"\nWARNING: {fb}/{len(df)} queries fell back to rules (LLM call failed). "
              f"The accuracy below is PARTLY NOT the LLM's. "
              f"Last error: {getattr(clf._llm, 'last_error', None) or getattr(clf, 'last_error', 'unknown')}")

    acc = accuracy_score(df.label, df.pred)
    p, r, f, s = precision_recall_fscore_support(df.label, df.pred, labels=LABELS, zero_division=0)
    per = pd.DataFrame({"intent": LABELS, "precision": p, "recall": r, "f1": f, "support": s})
    cm = pd.DataFrame(confusion_matrix(df.label, df.pred, labels=LABELS), index=LABELS, columns=LABELS)

    print(f"\nBackend: {a.backend} | split: {a.split} | n={len(df)} | data: {a.data}")
    print(f"Accuracy: {acc:.1%}  (target {TARGET:.0%}) -> {'PASS' if acc >= TARGET else 'FAIL'}")
    print(f"Routing correct: {df.routing_ok.mean():.1%}")
    print("\n" + classification_report(df.label, df.pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):\n", cm.to_string())

    dom = None
    if "domain" in df.columns and (df["domain"] != "").any():
        dom = (df.groupby("domain").agg(rows=("correct", "size"), correct=("correct", "sum"),
                                        accuracy=("correct", "mean"), routing=("routing_ok", "mean"))
               .reset_index().sort_values(["accuracy", "domain"]))
        print("\nAccuracy by domain (weakest first):\n" + dom.round(3).to_string(index=False))
    amb = df[df["ambiguous_term"] != ""] if "ambiguous_term" in df.columns else df.iloc[0:0]
    if len(amb):
        print(f"\nShared-word rows (same word in several domains): "
              f"{int(amb.correct.sum())}/{len(amb)} correct ({amb.correct.mean():.1%})")

    dacc = None
    if "expected_domain" in df.columns and (df["expected_domain"] != "").any():
        dsub = df[df["expected_domain"] != ""]
        dacc = float((dsub["expected_domain"] == dsub["pred_domain"]).mean())
        print(f"\nDomain accuracy (informational, not part of the 90% target): {dacc:.1%} on {len(dsub)} rows")
        dwrong = dsub[dsub["expected_domain"] != dsub["pred_domain"]]
        if len(dwrong):
            print(dwrong[["id", "query", "expected_domain", "pred_domain"]].head(15).to_string(index=False))

    wrong = df[~df.correct]
    pairs = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False).head(5)
    rep_pairs = [((t, p_), int(n)) for (t, p_), n in pairs.items()]
    print(f"\nMisclassified ({len(wrong)}):")
    print(wrong[["id", "query", "label", "pred", "confidence"]].to_string(index=False) if len(wrong) else "none")

    prefix = a.out or f"results_{a.backend}_{a.split}"
    routing = (df.groupby("label").routing_ok.agg(["mean", "sum", "count"])
               .rename(columns={"mean": "routing_accuracy", "sum": "routed_correctly", "count": "rows"})
               .reset_index())
    with pd.ExcelWriter(f"{prefix}.xlsx") as xw:
        df.drop(columns=[c for c in ("split", "difficulty") if c in df.columns]).to_excel(
            xw, sheet_name="predictions", index=False)
        per.to_excel(xw, sheet_name="per_intent", index=False)
        cm.to_excel(xw, sheet_name="confusion")
        routing.to_excel(xw, sheet_name="routing", index=False)
        if dom is not None:
            dom.to_excel(xw, sheet_name="by_domain", index=False)
    write_report(f"{prefix}.md", df, a.backend, a.split, acc, per, cm, rep_pairs)
    if dom is not None:
        with open(f"{prefix}.md", "a", encoding="utf-8") as fh:
            fh.write("\n## Accuracy by domain\n\n" + dom.round(3).to_markdown(index=False) + "\n")
            if len(amb):
                fh.write(f"\nShared-word rows: {int(amb.correct.sum())}/{len(amb)} correct "
                         f"({amb.correct.mean():.1%})\n")
    if dacc is not None:
        with open(f"{prefix}.md", "a", encoding="utf-8") as fh:
            fh.write(f"\nDomain accuracy (informational): {dacc:.1%} on {len(dsub)} rows\n")
    print(f"\nSaved {prefix}.xlsx and {prefix}.md")

    if a.split in ("test", "all") and a.backend != "rules":
        if fb:
            print(f"\nNOTE: {fb} rows fell back to rules, so this TEST result is NOT valid and the "
                  f"one-time lock was NOT set. Fix the cause (rate limit/timeouts) and rerun TEST.")
        else:
            lock.write_text(f"{pd.Timestamp.now():%Y-%m-%d %H:%M} acc={acc:.3f}")


if __name__ == "__main__":
    main()
