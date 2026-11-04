"""Write ironman_testset_v2.xlsx / .csv.

The visible sheet has NO split or difficulty columns. They are kept out of sight but still used by the code
(DEV/TEST runs, DEV-only few-shot examples): hidden sheet "_splits" in the xlsx, "<name>.splits.csv" next to the csv.
"""
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from testset_v2_data import rows

full = pd.DataFrame(list(rows()))
assert full["query"].is_unique, full[full["query"].duplicated(keep=False)]
cols = ["id", "query", "label", "domain", "ambiguous_term", "expected_route", "expected_domain"]
df = full[cols]
splits = full[["id", "split", "difficulty"]]

df.to_csv("ironman_testset_v2.csv", index=False)
splits.to_csv("ironman_testset_v2.splits.csv", index=False)

summary = df.groupby("label").size().rename("total").reset_index()
domains = df.groupby("domain").size().rename("total").reset_index()
terms = (df[df.ambiguous_term != ""].assign(term=lambda d: d.ambiguous_term.str.split(";"))
         .explode("term").groupby("term").agg(rows=("id", "count"),
                                              intents=("label", lambda s: ", ".join(sorted(set(s))))).reset_index())

with pd.ExcelWriter("ironman_testset_v2.xlsx", engine="openpyxl") as xw:
    df.to_excel(xw, sheet_name="testset", index=False)
    summary.to_excel(xw, sheet_name="summary", index=False)
    domains.to_excel(xw, sheet_name="domains", index=False)
    terms.to_excel(xw, sheet_name="shared_words", index=False)
    splits.to_excel(xw, sheet_name="_splits", index=False)
    for name, frame in (("testset", df), ("summary", summary), ("domains", domains), ("shared_words", terms)):
        ws = xw.sheets[name]
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="305496")
            c.alignment = Alignment(horizontal="center")
        for i, col in enumerate(frame.columns, 1):
            ws.column_dimensions[get_column_letter(i)].width = min(
                max(len(str(col)), *(len(str(v)) for v in frame[col])) + 2, 90)
        ws.freeze_panes = "A2"
    xw.sheets["_splits"].sheet_state = "hidden"

print(summary.to_string(index=False))
print(domains.to_string(index=False))
print("rows:", len(df), "| columns:", list(df.columns), "| DEV/TEST kept in hidden _splits:",
      full.split.value_counts().to_dict())
