from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, MultipleLocator



def plot_metric_vs_round_for_clients(
    csv_3_clients: str,
    csv_5_clients: str,
    csv_7_clients: str,
    metric: str,
    out_path: str,
    *,
    round_col: str = "round",
    title: str | None = None,
    y_label: str | None = None,
    y_lim: tuple[float, float] | None = None,
    labels: tuple[str, str, str] = ("MLPNet", "DCN-Lite", "TabResNet"),
    dpi: int = 300,
) -> None:
    """
    Vẽ biểu đồ đường (3 đường) so sánh metric theo round cho 3/5/7 clients.

    Parameters
    ----------
    csv_3_clients, csv_5_clients, csv_7_clients : str
        Đường dẫn file .csv
    metric : str
        Tên cột metric cần vẽ (global_test_acc, f1_weighted, f1_macro, ...)
    out_path : str
        Đường dẫn file ảnh output (.png, .pdf, ...)
    round_col : str
        Tên cột round (mặc định "round")
    title : str | None
        Tiêu đề biểu đồ
    y_label : str | None
        Nhãn trục Y hiển thị đẹp (ví dụ "Accuracy", "F1-score")
    y_lim : tuple(float, float) | None
        Giới hạn trục Y, ví dụ (0.88, 0.93)
    labels : tuple
        Nhãn cho 3 đường (3/5/7 clients)
    dpi : int
        Độ phân giải ảnh (mặc định 300 – chuẩn luận văn)
    """

    def _load_and_validate(csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

        df = pd.read_csv(csv_path)

        if round_col not in df.columns:
            raise ValueError(f"{csv_path} thiếu cột '{round_col}'")

        if metric not in df.columns:
            raise ValueError(f"{csv_path} thiếu cột metric '{metric}'")

        df = df.sort_values(by=round_col).reset_index(drop=True)
        df[round_col] = pd.to_numeric(df[round_col], errors="coerce")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        return df.dropna(subset=[round_col, metric])

    df3 = _load_and_validate(csv_3_clients)
    df5 = _load_and_validate(csv_5_clients)
    df7 = _load_and_validate(csv_7_clients)

    if title is None:
        title = "Sự phụ thuộc của F1-macro của 3 mô hình"

    if y_label is None:
        y_label = metric

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(df3[round_col], df3[metric], marker="o", label=labels[0])
    plt.plot(df5[round_col], df5[metric], marker="o", label=labels[1])
    plt.plot(df7[round_col], df7[metric], marker="o", label=labels[2])

    plt.xlabel("Số round")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if y_lim is not None:
        plt.ylim(y_lim)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    plot_metric_vs_round_for_clients(
        csv_3_clients="outputs\exp_mlp_non-iid_0.3_N7_v4\global_test_per_round.csv",
        csv_5_clients="outputs\exp_dcn-lite_non-iid_0.3_N7_v4\global_test_per_round.csv",
        csv_7_clients="outputs\exp_tab-res-net_non-iid_0.3_N7_v4\global_test_per_round.csv",
        metric="f1_macro",
        y_label="F1-macro",
        y_lim=(0.80, 0.95),
        out_path="charts_outputs/figs/non-iid-0.3-20-round-v4.png"
    )

if __name__ == "__main__":
    main()

