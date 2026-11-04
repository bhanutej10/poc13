"""Load a labeled test set from .xlsx or .csv and normalise its columns.

Accepts your IRONMAN workbook as-is. Column names are matched case-insensitively
against the aliases below, so you don't have to rename anything:

  query     : query | question | text | user_query | utterance
  label     : intent | label | expected_intent | true_intent | gold
  split     : split | set | dataset            (DEV / TEST; heldout is treated as TEST)
  route     : expected_route | route | expected_handler
  id        : id | qid | query_id | #           (optional, generated if missing)
  difficulty: difficulty | type | case_type     (optional)

Set TESTSET_PATH to point the classifier's few-shot examples (DEV rows only) at the same file.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd

from intents import Intent

ALIASES = {
    "query": ["query", "question", "text", "user_query", "utterance", "user query"],
    "label": ["intent", "label", "expected_intent", "true_intent", "gold", "expected intent"],
    "split": ["split", "set", "dataset", "data_split"],
    "expected_route": ["expected_route", "route", "expected_handler", "expected route"],
    "id": ["id", "qid", "query_id", "#", "no", "s.no"],
    "difficulty": ["difficulty", "case_type", "type", "lookalike"],
    "domain": ["domain", "department", "function", "business_area"],
    "ambiguous_term": ["ambiguous_term", "shared_word", "shared_term"],
    "expected_domain": ["expected_domain", "true_domain", "gold_domain"],
}

DEFAULT_PATH = os.getenv("TESTSET_PATH", "testset.csv")


def _norm_label(v) -> str:
    return re.sub(r"[\s\-/]+", "_", str(v).strip()).upper()


def _norm_split(v) -> str:
    s = str(v).strip().lower()
    if s in ("dev", "train", "tuning", "development"):
        return "dev"
    if s in ("test", "heldout", "held-out", "held_out", "holdout", "hold-out"):
        return "test"
    return s


def _pick(df: pd.DataFrame, key: str) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for a in ALIASES[key]:
        if a in lowered:
            return lowered[a]
    return None


def load(path: Optional[str] = None, sheet: Optional[str] = None) -> pd.DataFrame:
    path = path or DEFAULT_PATH
    side = None   # split/difficulty kept out of the visible sheet: hidden '_splits' sheet or <name>.splits.csv
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test set not found: {path!r} (looked in {os.getcwd()}). Copy your workbook there or pass --data /full/path.xlsx")
        xl = pd.ExcelFile(path)
        if "_splits" in xl.sheet_names:
            side = pd.read_excel(xl, sheet_name="_splits")
        # use the requested sheet, else the first sheet that has a query-like column
        names = [sheet] if sheet else xl.sheet_names
        raw = None
        for n in names:
            cand = pd.read_excel(xl, sheet_name=n)
            if _pick(cand, "query") is not None and _pick(cand, "label") is not None:
                raw = cand
                break
        if raw is None:
            raise ValueError(
                f"No sheet in {path} has both a query column and an intent column. "
                f"Sheets: {xl.sheet_names}. Expected aliases: {ALIASES['query']} / {ALIASES['label']}"
            )
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test set not found: {path!r} (looked in {os.getcwd()}). Copy your workbook there or pass --data /full/path.xlsx")
        raw = pd.read_csv(path)
        sidecar = os.path.splitext(path)[0] + ".splits.csv"
        if os.path.exists(sidecar):
            side = pd.read_csv(sidecar)

    cols = {k: _pick(raw, k) for k in ALIASES}
    if cols["query"] is None or cols["label"] is None:
        raise ValueError(f"Need a query column and an intent column. Found: {list(raw.columns)}")

    df = pd.DataFrame()
    df["query"] = raw[cols["query"]].astype(str).str.strip()
    df["label"] = raw[cols["label"]].map(_norm_label)
    df["id"] = raw[cols["id"]].astype(str) if cols["id"] else [f"Q{i+1:03d}" for i in range(len(raw))]
    df["split"] = raw[cols["split"]].map(_norm_split) if cols["split"] else "test"
    df["expected_route"] = (
        raw[cols["expected_route"]].astype(str).str.strip() if cols["expected_route"] else None
    )
    df["difficulty"] = raw[cols["difficulty"]].astype(str) if cols["difficulty"] else ""
    df["domain"] = raw[cols["domain"]].fillna("").astype(str).str.strip() if cols["domain"] else ""
    df["expected_domain"] = (raw[cols["expected_domain"]].fillna("").astype(str).str.strip()
                             if cols["expected_domain"] else "")
    df["ambiguous_term"] = (raw[cols["ambiguous_term"]].fillna("").astype(str).str.strip()
                            if cols["ambiguous_term"] else "")

    if cols["split"] is None and side is not None:       # merge split/difficulty by id
        sc = {str(k).strip().lower(): k for k in side.columns}
        if "id" in sc:
            ids = side[sc["id"]].astype(str)
            if "split" in sc:
                df["split"] = df["id"].map(dict(zip(ids, side[sc["split"]].map(_norm_split)))).fillna("test")
            if "difficulty" in sc and cols["difficulty"] is None:
                df["difficulty"] = df["id"].map(dict(zip(ids, side[sc["difficulty"]].fillna("").astype(str)))).fillna("")

    df = df[df["query"].str.len() > 0].reset_index(drop=True)
    valid = {i.value for i in Intent}
    bad = sorted(set(df["label"]) - valid)
    if bad:
        raise ValueError(f"Unknown intent labels in {path}: {bad}. Allowed: {sorted(valid)}")
    return df


def dev_examples(path: Optional[str] = None, per_intent: Optional[int] = None) -> list[tuple[str, str]]:
    """(query, label) few-shot pairs from DEV rows only.

    Lookalike/hard rows come first, then rows are spread across business domains so one function
    (say IT) does not dominate the prompt. Labels are interleaved to avoid a recency bias.
    per_intent: argument > FEWSHOT_PER_INTENT env > 6 if the set has a domain column, else 3.
    """
    df = load(path)
    df = df[df["split"] == "dev"].copy()
    if per_intent is None:
        env = os.getenv("FEWSHOT_PER_INTENT")
        per_intent = int(env) if env else (6 if (df["domain"] != "").any() else 3)
    df["_hard"] = df["difficulty"].str.lower().str.contains("hard|lookalike|yes|true")
    per_label: list[list[tuple[str, str]]] = []
    for label, g in df.groupby("label", sort=False):
        ordered = pd.concat([g[g["_hard"]], g[~g["_hard"]]])
        picked: list[str] = []
        used: set[str] = set()
        for _, r in ordered.iterrows():          # pass 1: one row per domain
            if len(picked) >= per_intent:
                break
            if r["domain"] and r["domain"] in used:
                continue
            picked.append(r["query"])
            used.add(r["domain"])
        for _, r in ordered.iterrows():          # pass 2: fill up from whatever is left
            if len(picked) >= per_intent:
                break
            if r["query"] not in picked:
                picked.append(r["query"])
        per_label.append([(q, label) for q in picked])
    out: list[tuple[str, str]] = []
    for i in range(max((len(x) for x in per_label), default=0)):
        out += [x[i] for x in per_label if i < len(x)]
    return out


def dev_domains(path: Optional[str] = None) -> dict[str, str]:
    """query -> expected domain for DEV rows that have one (used to show the domain in few-shot examples)."""
    df = load(path)
    df = df[(df["split"] == "dev") & (df["expected_domain"] != "")]
    return dict(zip(df["query"], df["expected_domain"]))
