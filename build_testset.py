"""Write the labeled test set to testset.xlsx and testset.csv."""
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from testset_data import rows

df = pd.DataFrame(list(rows()))
df.to_csv("testset.csv", index=False)

with pd.ExcelWriter("testset.xlsx", engine="openpyxl") as xw:
    df.to_excel(xw, sheet_name="testset", index=False)
    summary = (
        df.groupby(["label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda t: t.sum(axis=1))
        .reset_index()
    )
    summary.to_excel(xw, sheet_name="summary", index=False)

    for name, frame in (("testset", df), ("summary", summary)):
        ws = xw.sheets[name]
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="305496")
            c.alignment = Alignment(horizontal="center")
        for i, col in enumerate(frame.columns, 1):
            width = max(len(str(col)), *(len(str(v)) for v in frame[col])) + 2
            ws.column_dimensions[get_column_letter(i)].width = min(width, 80)
        ws.freeze_panes = "A2"

print(df.groupby(["label", "split"]).size().unstack())
print("hard:", (df.difficulty == "hard").sum(), "total:", len(df))
