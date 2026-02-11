import pandas as pd

# ===== CONFIG =====
INPUT_CSV = "data/test_final.csv"
OUTPUT_CSV = "data/infer_bench.csv"
N_SAMPLES = 100_000
RANDOM_SEED = 42
# ==================

df = pd.read_csv(INPUT_CSV)

# nếu test nhỏ hơn 10k thì lấy toàn bộ
n = min(N_SAMPLES, len(df))

infer_df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

infer_df.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] Saved infer bench CSV: {OUTPUT_CSV}")
print(f"[OK] Samples: {len(infer_df)}")
print(f"[OK] Columns: {infer_df.shape[1]}")
