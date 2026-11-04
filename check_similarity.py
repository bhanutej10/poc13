"""Measure how well the paraphrase matcher separates real rephrasings from lookalikes.

  python check_similarity.py                       # lexical matcher (no downloads)
  python check_similarity.py --backend embedding   # needs: pip install fastembed
  python check_similarity.py --backend embedding --sweep    # try several thresholds, pick yours

File columns: original, other, should_match (1/0), tier (easy|hard|intent|detail), note.
A WRONG MATCH (should_match=0 but matched) returns the wrong stored answer: keep that at 0.
A missed paraphrase only costs one normal LLM call.
Add your own pairs to paraphrase_pairs.csv: the shipped ones were written together with the matcher.
"""
import argparse

import pandas as pd

from similarity import SimilarityMatcher, EmbeddingMatcher, LexicalMatcher, guard_ok


def run(df, matcher):
    rows = []
    for _, r in df.iterrows():
        hit = matcher.best(r["other"], [r["original"]])
        rows.append({**r, "matched": hit is not None,
                     "score": round(matcher.scorer.similarity(r["other"], r["original"]), 3),
                     "guard": guard_ok(r["other"], r["original"])})
    return pd.DataFrame(rows)


def summarize(res):
    pos, neg = res[res.should_match == 1], res[res.should_match == 0]
    wrong = neg[neg.matched]
    print(f"  wrong matches (lookalikes that matched): {len(wrong)}/{len(neg)}")
    for tier in ("easy", "hard"):
        t = pos[pos.tier == tier]
        print(f"  paraphrases matched, {tier:4}: {int(t.matched.sum())}/{len(t)}")
    return wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paraphrase_pairs.csv")
    ap.add_argument("--backend", default="lexical", choices=["lexical", "embedding"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    scorer = EmbeddingMatcher() if a.backend == "embedding" else LexicalMatcher()
    ths = [0.80, 0.84, 0.88, 0.92, 0.95] if a.sweep else [a.threshold or scorer.default_threshold]
    for th in ths:
        print(f"backend={a.backend} threshold={th}")
        res = run(df, SimilarityMatcher(scorer, th))
        wrong = summarize(res)
        if len(wrong):
            print(wrong[["original", "other", "score", "note"]].to_string(index=False))
    print("\nmissed paraphrases at the last threshold:")
    miss = res[(res.should_match == 1) & (~res.matched)]
    print(miss[["tier", "original", "other", "score", "guard"]].to_string(index=False) if len(miss) else "none")
    print("\nPick the LOWEST threshold that keeps wrong matches at 0, then set SIMILARITY_THRESHOLD to it.")


if __name__ == "__main__":
    main()
