"""Remove the split and difficulty columns from a workbook's visible sheet without losing them.

  python hide_columns.py ironman_testset.xlsx            -> ironman_testset_clean.xlsx

The new workbook has one visible sheet (no split/difficulty) and a hidden "_splits" sheet (id, split, difficulty),
which evaluate.py and the few-shot selection still read, so --split dev/test keeps working.
Rows need an id column; one is added (Q001, Q002 ...) if missing.
"""
import sys

import pandas as pd

import dataset


def main(src, dst=None):
    xl = pd.ExcelFile(src)
    raw = next((pd.read_excel(xl, sheet_name=n) for n in xl.sheet_names
                if dataset._pick(pd.read_excel(xl, sheet_name=n), "query") is not None
                and dataset._pick(pd.read_excel(xl, sheet_name=n), "label") is not None), None)
    if raw is None:
        sys.exit("No sheet with a query column and an intent column.")
    id_col, sp_col, df_col = (dataset._pick(raw, k) for k in ("id", "split", "difficulty"))
    if id_col is None:
        raw.insert(0, "id", [f"Q{i + 1:03d}" for i in range(len(raw))])
        id_col = "id"
    side = pd.DataFrame({"id": raw[id_col].astype(str),
                         "split": raw[sp_col] if sp_col else "test",
                         "difficulty": raw[df_col] if df_col else ""})
    visible = raw.drop(columns=[c for c in (sp_col, df_col) if c])
    dst = dst or src.rsplit(".", 1)[0] + "_clean.xlsx"
    with pd.ExcelWriter(dst, engine="openpyxl") as xw:
        visible.to_excel(xw, sheet_name="testset", index=False)
        side.to_excel(xw, sheet_name="_splits", index=False)
        xw.sheets["_splits"].sheet_state = "hidden"
    print(f"wrote {dst}: visible columns {list(visible.columns)}; split/difficulty kept in hidden sheet _splits")


if __name__ == "__main__":
    main(*sys.argv[1:3]) if len(sys.argv) > 1 else sys.exit(__doc__)
