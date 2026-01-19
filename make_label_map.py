import json, pandas as pd

df = pd.read_csv("data/train_final.csv")
label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])
labels = sorted(df[label_col].astype(str).unique().tolist())
m = {lab:i for i,lab in enumerate(labels)}

with open("data/label_map.json","w",encoding="utf-8") as f:
    json.dump({"labels":labels,"map":m}, f, ensure_ascii=False, indent=2)

print("saved data/label_map.json, num_classes=", len(labels))
