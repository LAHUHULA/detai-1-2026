import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN = DATA_DIR / "train_final.csv"

OUT_BASE = DATA_DIR / "clients"
N_CLIENTS_LIST = [10]   # sẽ tạo iid_n3, iid_n5, iid_n7
ALPHAS = [0.3]          # sẽ tạo dirichlet_a0.3_n3... và a0.5...
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

def split_iid(df, n_clients: int):
    rng = np.random.RandomState(SEED)
    idx_all = rng.permutation(len(df))
    chunks = np.array_split(idx_all, n_clients)

    out_dir = OUT_BASE / f"iid_n{n_clients}"
    for i, idxs in enumerate(chunks):
        write_client_csv(df, idxs, out_dir / f"client_{i}.csv")

    print(f"[OK] IID splits written to: {out_dir}")
    sizes = [len(x) for x in chunks]
    print("Client sizes:", sizes)
    return out_dir


def split_dirichlet(df, y_raw, alpha: float, n_clients: int):
    rng = np.random.RandomState(SEED)
    classes = np.unique(y_raw)

    idx_by_class = {c: np.where(y_raw == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        idx_c = idx_by_class[c]
        n_c = len(idx_c)
        if n_c == 0:
            continue

        props = rng.dirichlet(alpha * np.ones(n_clients))
        counts = (props * n_c).astype(int)

        diff = n_c - counts.sum()
        while diff > 0:
            counts[rng.randint(0, n_clients)] += 1
            diff -= 1
        while diff < 0:
            j = rng.randint(0, n_clients)
            if counts[j] > 0:
                counts[j] -= 1
                diff += 1

        start = 0
        for cid in range(n_clients):
            end = start + counts[cid]
            if end > start:
                client_indices[cid].extend(idx_c[start:end].tolist())
            start = end

    # avoid empty clients by stealing
    empty = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
    if empty:
        donors = sorted(range(n_clients), key=lambda k: len(client_indices[k]), reverse=True)
        for ec in empty:
            donor = None
            for d in donors:
                if len(client_indices[d]) > 1:
                    donor = d
                    break
            if donor is None:
                raise RuntimeError("Cannot fix empty client; reduce clients or increase alpha.")
            client_indices[ec].append(client_indices[donor].pop())

    out_dir = OUT_BASE / f"dirichlet_a{alpha}_n{n_clients}"
    for i in range(n_clients):
        rng.shuffle(client_indices[i])
        write_client_csv(df, np.array(client_indices[i], dtype=int), out_dir / f"client_{i}.csv")

    print(f"[OK] Dirichlet(alpha={alpha}) splits written to: {out_dir}")
    sizes = [len(x) for x in client_indices]
    print("Client sizes:", sizes)
    return out_dir


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


def show_partition_distribution(
    out_dir: str | Path,
    label_col: str = "label",
    num_partitions: int = 7,
    top_k_classes: int | None = None,
    plot: bool = True,
    save_path: str | Path | None = None,
    normalize: str = "percent",  # "percent" hoặc "count"
):
    """
    Vẽ phân phối nhãn theo từng client dựa trên các file client_*.csv trong out_dir.

    Args:
      out_dir: thư mục chứa client_0.csv ... client_{k-1}.csv
      label_col: tên cột nhãn trong csv
      num_partitions: số client
      top_k_classes: nếu set (vd 13), chỉ giữ top K lớp theo global count.
                    nếu nhiều hơn K lớp, sẽ gộp phần còn lại vào "Other".
      plot: True để hiển thị
      save_path: nếu set, sẽ lưu hình (png) vào path này
      normalize: "percent" hoặc "count"
    Returns:
      counts_df, pct_df
    """
    out_dir = Path(out_dir)
    paths = [out_dir / f"client_{i}.csv" for i in range(num_partitions)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing[:3]}{'...' if len(missing)>3 else ''}")

    # counts per client
    counts_list = []
    for i, p in enumerate(paths):
        cdf = pd.read_csv(p)
        if label_col not in cdf.columns:
            raise ValueError(f"label_col='{label_col}' not found in {p}. Columns={list(cdf.columns)}")
        s = cdf[label_col].astype(str).value_counts()
        s.name = f"client_{i}"
        counts_list.append(s)

    counts_df = pd.concat(counts_list, axis=1).fillna(0).astype(int)

    # global class order
    global_counts = counts_df.sum(axis=1).sort_values(ascending=False)

    # top_k handling
    if top_k_classes is not None and top_k_classes > 0 and len(global_counts) > top_k_classes:
        top_classes = global_counts.index[:top_k_classes]
        other_classes = global_counts.index[top_k_classes:]
        # build reduced counts
        reduced = counts_df.loc[top_classes].copy()
        reduced.loc["Other"] = counts_df.loc[other_classes].sum(axis=0)
        counts_df = reduced
        global_counts = counts_df.sum(axis=1).sort_values(ascending=False)
    else:
        # reorder by global frequency
        counts_df = counts_df.loc[global_counts.index]

    # percent per client
    pct_df = counts_df.div(counts_df.sum(axis=0), axis=1).fillna(0.0) * 100.0

    if not plot and save_path is None:
        return counts_df, pct_df.round(2)

    # ---------- Plot ----------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 0.30], hspace=0.25)

    # 1) stacked bar
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(num_partitions)
    bottoms = np.zeros(num_partitions, dtype=float)

    if normalize == "count":
        plot_df = counts_df
        ylabel = "Count"
        title = f"Partition label distribution (stacked counts) - {out_dir.name}"
    else:
        plot_df = pct_df
        ylabel = "Percent (%)"
        title = f"Partition label distribution (stacked %) - {out_dir.name}"

    for cls in plot_df.index:
        vals = plot_df.loc[cls].to_numpy(dtype=float)
        ax1.bar(x, vals, bottom=bottoms, label=str(cls))
        bottoms += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"client_{i}" for i in range(num_partitions)], rotation=0)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    # 2) heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    heat = pct_df.to_numpy(dtype=float)
    im = ax2.imshow(heat, aspect="auto")  # ax2.imshow, không dùng plt.imshow

    ax2.set_yticks(np.arange(pct_df.shape[0]))
    ax2.set_yticklabels([str(i) for i in pct_df.index])
    ax2.set_xticks(np.arange(num_partitions))
    ax2.set_xticklabels([f"client_{i}" for i in range(num_partitions)], rotation=0)
    ax2.set_title("Heatmap: class % by client")

    # đảm bảo tick x chỉ ở dưới
    ax2.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    cbar.set_label("%")

    # annotate (>=10%)
    for r in range(pct_df.shape[0]):
        for c in range(num_partitions):
            v = heat[r, c]
            if v >= 10:
                ax2.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=8)

    # 3) legend riêng (không đè heatmap)
    ax_leg = fig.add_subplot(gs[2, 0])
    ax_leg.set_axis_off()   # tắt hoàn toàn axes -> không còn 0..1
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    handles, labels = ax1.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncol=4, frameon=False, fontsize=9)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


    return counts_df, pct_df.round(2)


if __name__ == "__main__":
    df, label_col, y_raw = load_train()

    # 1) IID for N=3,5,7
    # for n in N_CLIENTS_LIST:
    #     iid_dir = split_iid(df, n_clients=n)
    #     report_splits(df, label_col, iid_dir, n_clients=n, save_csv=True)

    #     # optional plot
    #     show_partition_distribution(
    #         out_dir=iid_dir,
    #         label_col=label_col,
    #         num_partitions=n,
    #         top_k_classes=13,
    #         plot=True,
    #         save_path=iid_dir / "_partition_distribution.png",
    #         normalize="percent",
    #     )

    # 2) Dirichlet for each alpha and each N
    for alpha in ALPHAS:
        for n in N_CLIENTS_LIST:
            non_iid_dir = split_dirichlet(df, y_raw, alpha=alpha, n_clients=n)
            report_splits(df, label_col, non_iid_dir, n_clients=n, save_csv=True)

            # optional plot
            show_partition_distribution(
                out_dir=non_iid_dir,
                label_col=label_col,
                num_partitions=n,
                top_k_classes=13,
                plot=True,
                save_path=non_iid_dir / "_partition_distribution.png",
                normalize="percent",
            )

    print("\nDone.")

