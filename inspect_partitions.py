# from ddos_attack.task import show_partition_distribution

def show_partition_distribution(
    mode: str,
    num_partitions: int,
    alpha: float = 0.5,
    seed: int = 42,
    top_k_classes: int | None = None,
    plot: bool = True,
):
    """
    Hiển thị phân phối nhãn theo từng client:
    - counts: số mẫu mỗi lớp trên mỗi client
    - ratios: tỉ lệ (%)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    _ensure_data_loaded()
    y = _train_labels
    num_classes = int(y.max()) + 1

    # build/get partitions (indices)
    key = (mode, float(alpha), int(num_partitions))
    if key not in _partitions_cache:
        _partitions_cache[key] = _build_partitions(mode, num_partitions, alpha=alpha, seed=seed)

    parts = _partitions_cache[key]  # list[np.ndarray indices]

    # counts matrix: [num_partitions, num_classes]
    counts = np.zeros((num_partitions, num_classes), dtype=np.int64)
    sizes = np.zeros((num_partitions,), dtype=np.int64)

    for cid in range(num_partitions):
        idxs = np.asarray(parts[cid], dtype=np.int64)
        sizes[cid] = len(idxs)
        if len(idxs) > 0:
            counts[cid] = np.bincount(y[idxs], minlength=num_classes)

    # DataFrame for display
    df_counts = pd.DataFrame(
        counts,
        index=[f"client_{i}" for i in range(num_partitions)],
        columns=[f"class_{j}" for j in range(num_classes)],
    )
    df_counts.insert(0, "num_samples", sizes)

    # ratios
    denom = np.maximum(sizes.reshape(-1, 1), 1)
    ratios = counts / denom
    df_ratios = pd.DataFrame(
        ratios,
        index=df_counts.index,
        columns=df_counts.columns[1:],  # class columns only
    )

    # optional: show only top_k most frequent classes globally (for readability)
    if top_k_classes is not None and top_k_classes < num_classes:
        global_counts = counts.sum(axis=0)
        top_idx = np.argsort(global_counts)[::-1][:top_k_classes]
        keep_cols = ["num_samples"] + [f"class_{j}" for j in top_idx]
        df_counts = df_counts[keep_cols]
        df_ratios = df_ratios[[f"class_{j}" for j in top_idx]]
        counts_to_plot = counts[:, top_idx]
        class_labels = [f"class_{j}" for j in top_idx]
    else:
        counts_to_plot = counts
        class_labels = [f"class_{j}" for j in range(num_classes)]

    # Print summary
    print(f"\n=== Partition distribution | mode={mode} | alpha={alpha} | num_partitions={num_partitions} ===")
    print(df_counts)

    # Basic non-IID indicator: per-client entropy (higher => more mixed)
    eps = 1e-12
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    print("\nEntropy per client (higher => more mixed):")
    for i, e in enumerate(entropy):
        print(f"  client_{i}: {e:.4f}  (n={sizes[i]})")

    if plot:
        # Stacked bar chart (counts)
        plt.figure()
        bottom = np.zeros((num_partitions,), dtype=np.int64)
        for j, lab in enumerate(class_labels):
            plt.bar(np.arange(num_partitions), counts_to_plot[:, j], bottom=bottom, label=lab)
            bottom += counts_to_plot[:, j]
        plt.title(f"Dirichlet partition (mode={mode}, alpha={alpha}) - counts")
        plt.xlabel("Client")
        plt.ylabel("Number of samples")
        plt.xticks(np.arange(num_partitions), [str(i) for i in range(num_partitions)])
        if len(class_labels) <= 15:
            plt.legend()
        plt.tight_layout()
        plt.show()

        # Heatmap (ratios)
        plt.figure()
        ratios_to_plot = counts_to_plot / np.maximum(counts_to_plot.sum(axis=1, keepdims=True), 1)
        plt.imshow(ratios_to_plot, aspect="auto")
        plt.title(f"Dirichlet partition (mode={mode}, alpha={alpha}) - class ratios")
        plt.xlabel("Class")
        plt.ylabel("Client")
        plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90)
        plt.yticks(np.arange(num_partitions), [f"c{i}" for i in range(num_partitions)])
        plt.colorbar(label="ratio")
        plt.tight_layout()
        plt.show()

    return df_counts, df_ratios

if __name__ == "__main__":
    # Bạn chỉnh theo cấu hình bạn đang chạy
    num_clients = 5
    alpha = 0.5

    show_partition_distribution(
        mode="dirichlet",
        num_partitions=num_clients,
        alpha=alpha,
        seed=42,
        top_k_classes=13,   # bạn có 13 lớp
        plot=True,
    )
