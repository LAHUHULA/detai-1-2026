import json
import pandas as pd

CSV = "data/train_final.csv"
OUT = "data/label_map.json"

df = pd.read_csv(CSV)
label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])

labels = sorted(df[label_col].astype(str).unique().tolist())
label_map = {lab: i for i, lab in enumerate(labels)}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Saved:", OUT)
print("Num classes:", len(labels))
print(label_map)
