from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.ticker import MaxNLocator, NullLocator


def plot_metric_vs_round_for_clients(
    csv_3_clients: str | None,
    csv_5_clients: str | None,
    csv_7_clients: str | None,
    metric: str,
    out_path: str,
    *,
    round_col: str = "round",
    title: str | None = None,
    y_label: str | None = None,
    y_lim: tuple[float, float] | None = None,
    labels: tuple[str, str, str] = ("FedAvg", "FedProx-0", "FedProx-μ"),
    dpi: int = 300,
) -> None:

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })

    def _load_and_validate(csv_path: str | None) -> pd.DataFrame | None:
        if csv_path is None:
            return None

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
        title = "Metric vs Round"

    if y_label is None:
        y_label = metric

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    if df3 is not None:
        plt.plot(
            df3[round_col], df3[metric],
            linestyle=":",              
            linewidth=1.8,
            color="#ff7f0e",             
            label=labels[0]
        )

    if df5 is not None:
        plt.plot(
            df5[round_col], df5[metric],
            linestyle="--",              
            linewidth=1.8,
            color="#e377c2",             
            label=labels[1]
        )

    if df7 is not None:
        plt.plot(
            df7[round_col], df7[metric],
            linestyle="-",               
            linewidth=2.0,
            color="#17becf",             
            label=labels[2]
        )

    plt.xlabel("Số round")
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    if any(df is not None for df in (df3, df5, df7)):
        plt.legend()

    if y_lim is not None:
        plt.ylim(y_lim)

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MultipleLocator(0.02))

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(NullLocator())

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    plot_metric_vs_round_for_clients(
        csv_3_clients="outputs\exp_tab-res-net_non-iid-0.3_N10_fed_avg_ce-ft5_2_3_2026_v1\global_test_per_round.csv",   
        csv_5_clients="outputs\exp_tab-res-net_non-iid-0.3_N10_fed_avg_ce-ft7_2_3_2026_v1\global_test_per_round.csv",
        csv_7_clients="outputs\exp_tab-res-net_non-iid-0.3_N10_fed_prox-0.001_ce-ft7_2_3_2026_v1\global_test_per_round.csv",
        metric="round_train_loss",
        y_label="Round Train Loss",
        y_lim=(0.00, 0.5),
        out_path="charts_outputs/figs/non-iid-0.3-30-round-v1.png"
    )


if __name__ == "__main__":
    main()