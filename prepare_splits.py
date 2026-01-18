import os
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN = DATA_DIR / "train_final.csv"

OUT_BASE = DATA_DIR / "clients"
N_CLIENTS = 7
SEED = 42
EPS = 1e-12

def load_train():
    df = pd.read_csv(TRAIN)
    label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])
    y = df[label_col].astype(str).to_numpy()
    X = df.drop(columns=[label_col])
    return df, label_col, y

def write_client_csv(df, idxs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.iloc[idxs].to_csv(out_path, index=False)

def split_iid(df):
    rng = np.random.RandomState(SEED)
    idx_all = rng.permutation(len(df))
    chunks = np.array_split(idx_all, N_CLIENTS)
    out_dir = OUT_BASE / "iid"
    for i, idxs in enumerate(chunks):
        write_client_csv(df, idxs, out_dir / f"client_{i}.csv")
    print(f"[OK] IID splits written to: {out_dir}")

def split_dirichlet(df, y_raw, alpha: float):
    rng = np.random.RandomState(SEED)
    classes = np.unique(y_raw)

    idx_by_class = {c: np.where(y_raw == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(N_CLIENTS)]

    for c in classes:
        idx_c = idx_by_class[c]
        n_c = len(idx_c)
        if n_c == 0:
            continue

        props = rng.dirichlet(alpha * np.ones(N_CLIENTS))
        counts = (props * n_c).astype(int)

        diff = n_c - counts.sum()
        while diff > 0:
            counts[rng.randint(0, N_CLIENTS)] += 1
            diff -= 1
        while diff < 0:
            j = rng.randint(0, N_CLIENTS)
            if counts[j] > 0:
                counts[j] -= 1
                diff += 1

        start = 0
        for cid in range(N_CLIENTS):
            end = start + counts[cid]
            if end > start:
                client_indices[cid].extend(idx_c[start:end].tolist())
            start = end

    # avoid empty clients by stealing
    empty = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
    if empty:
        donors = sorted(range(N_CLIENTS), key=lambda k: len(client_indices[k]), reverse=True)
        for ec in empty:
            donor = None
            for d in donors:
                if len(client_indices[d]) > 1:
                    donor = d
                    break
            if donor is None:
                raise RuntimeError("Cannot fix empty client; reduce clients or increase alpha.")
            client_indices[ec].append(client_indices[donor].pop())

    out_dir = OUT_BASE / f"dirichlet_a{alpha}"
    for i in range(N_CLIENTS):
        rng.shuffle(client_indices[i])
        write_client_csv(df, np.array(client_indices[i], dtype=int), out_dir / f"client_{i}.csv")

    print(f"[OK] Dirichlet(alpha={alpha}) splits written to: {out_dir}")
    sizes = [len(x) for x in client_indices]
    print("Client sizes:", sizes)

def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) for a probability vector p."""
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (nats) between two prob vectors p, q."""
    p = np.clip(p, EPS, 1.0); p = p / p.sum()
    q = np.clip(q, EPS, 1.0); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))

def report_splits(
    df: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    n_clients: int,
    classes_order: list | None = None,
    save_csv: bool = True
):
    """
    Reads client_*.csv from out_dir, prints:
    - count table
    - % table
    - entropy per client
    - JS divergence vs global per client
    Also saves report CSVs if save_csv=True.
    """
    client_paths = [out_dir / f"client_{i}.csv" for i in range(n_clients)]
    for p in client_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    # Collect per-client label counts
    counts_list = []
    for i, p in enumerate(client_paths):
        cdf = pd.read_csv(p)
        y = cdf[label_col].astype(str)
        counts = y.value_counts()
        counts.name = f"client_{i}"
        counts_list.append(counts)

    counts_df = pd.concat(counts_list, axis=1).fillna(0).astype(int)

    # Fix class order
    if classes_order is None:
        classes_order = sorted(df[label_col].astype(str).unique().tolist())
    counts_df = counts_df.reindex(classes_order).fillna(0).astype(int)

    # Global distribution
    global_counts = df[label_col].astype(str).value_counts().reindex(classes_order).fillna(0).astype(int)
    global_p = (global_counts / global_counts.sum()).to_numpy(dtype=float)

    # Per-client %
    pct_df = counts_df.div(counts_df.sum(axis=0), axis=1).fillna(0.0)
    pct_df = (pct_df * 100.0).round(2)

    # Metrics per client
    metrics = []
    for col in counts_df.columns:
        p = (counts_df[col] / max(int(counts_df[col].sum()), 1)).to_numpy(dtype=float)
        ent = _entropy(p)
        js = _js_divergence(p, global_p)
        max_label_share = float(np.max(p)) if p.size else 0.0
        n_present = int(np.sum(counts_df[col].to_numpy() > 0))
        metrics.append({
            "client": col,
            "n_samples": int(counts_df[col].sum()),
            "n_labels_present": n_present,
            "max_label_share": round(max_label_share, 4),
            "entropy_nats": round(ent, 6),
            "js_divergence_vs_global_nats": round(js, 6),
        })

    metrics_df = pd.DataFrame(metrics).set_index("client")

    # Pretty print
    print("\n===== LABEL COUNT TABLE =====")
    print(counts_df)

    print("\n===== LABEL PERCENT TABLE (%) =====")
    print(pct_df)

    print("\n===== GLOBAL LABEL DISTRIBUTION (%) =====")
    global_pct = (global_counts / global_counts.sum() * 100.0).round(2)
    print(global_pct.to_string())

    print("\n===== METRICS (per client) =====")
    print(metrics_df.sort_index())

    if save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(out_dir / "_report_label_counts.csv")
        pct_df.to_csv(out_dir / "_report_label_percent.csv")
        metrics_df.to_csv(out_dir / "_report_metrics.csv")
        global_counts.to_frame("global_count").to_csv(out_dir / "_report_global_counts.csv")
        global_pct.to_frame("global_percent").to_csv(out_dir / "_report_global_percent.csv")
        print(f"\n[OK] Reports saved to: {out_dir}")

if __name__ == "__main__":
    df, label_col, y_raw = load_train()

    # split_iid(df)

    # iid_dir = OUT_BASE / "iid"
    # report_splits(df, label_col, iid_dir, N_CLIENTS)

    # split_dirichlet(df, y_raw, alpha=0.3)

    non_iid_dir = OUT_BASE / "dirichlet_a0.3"
    report_splits(df, label_col, non_iid_dir, N_CLIENTS)

    print("\nDone.")
