from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator


# ==== NEW ====
def _load_and_validate(csv_path: str | None, round_col: str, metric: str) -> pd.DataFrame | None:
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


# ==== NEW ====
def plot_grid_scenarios_models(
    data_map: dict,
    metric: str,
    out_path: str,
    *,
    round_col: str = "round",
    y_label: str = "Global Test Loss",
    y_lim: tuple[float, float] | None = None,
    labels=("FedAvg", "FedProx-0.01", "FedProx-0.05", "FedProx-0.1", "FedProx-0.5"),
    dpi: int = 300,
):

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9
    })

    scenarios = list(data_map.keys())
    models = list(next(iter(data_map.values())).keys())

    n_rows = len(models)      # 3
    n_cols = len(scenarios)   # 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 6),
        sharey=True
    )
    fig.text(
        0.04,
        0.5,
        y_label,
        rotation=90,
        va="center",
        fontsize=11
    )

    # ==== FIX nếu chỉ 1 hàng ====
    if n_rows == 1:
        axes = [axes]

    for r, model in enumerate(models):
        for c, scenario in enumerate(scenarios):

            ax = axes[r][c]

            csvs = data_map[scenario][model]

            df3 = _load_and_validate(csvs.get("fedavg"), round_col, metric)
            df5 = _load_and_validate(csvs.get("prox001"), round_col, metric)
            df7 = _load_and_validate(csvs.get("prox005"), round_col, metric)
            df9 = _load_and_validate(csvs.get("prox01"), round_col, metric)
            df11 = _load_and_validate(csvs.get("prox05"), round_col, metric)

            if df3 is not None:
                ax.plot(df3[round_col], df3[metric],
                        linestyle="--", linewidth=1.8, color="#0072B2", label=labels[0])

            if df5 is not None:
                ax.plot(df5[round_col], df5[metric],
                        linestyle="-", linewidth=1.8, color="#E69F00", label=labels[1])

            if df7 is not None:
                ax.plot(df7[round_col], df7[metric],
                        linestyle="-", linewidth=2.0, color="#e377c2", label=labels[2])

            if df9 is not None:
                ax.plot(df9[round_col], df9[metric],
                        linestyle="-", linewidth=2.0, color="#2ca02c", label=labels[3])

            if df11 is not None:
                ax.plot(df11[round_col], df11[metric],
                        linestyle="-.", linewidth=2.0, color="#009E73", label=labels[4])

            if r == 0:
                ax.set_title(scenario)

            if r == n_rows - 1:
                ax.set_xlabel("Số round")

            # if c == 0:
            #     ax.set_ylabel(model)

            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

            if y_lim is not None:
                ax.set_ylim(y_lim)
                ax.set_yticks([0.0, 0.4, 0.8, 1.2])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_minor_locator(NullLocator())

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for i, label in enumerate(models):
        fig.text(
            0.01,
            0.83 - i * 0.32,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

    handles, labels = axes[0][0].get_legend_handles_labels()

    plt.subplots_adjust(bottom=0.15)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=dpi)
    plt.close()


# ==== MODIFIED MAIN ====
def main():

    data_map = {

        "30% clients": {

            "LRNet-Lite": {
                "fedavg": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_avg_ce-ft3\global_test_per_round.csv",
                "prox001": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.01_ce-ft3\global_test_per_round.csv",
                # "prox005": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.05_ce-ft3\global_test_per_round.csv",
                "prox01": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.1_ce-ft3\global_test_per_round.csv",
                # "prox05": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.5_ce-ft3\global_test_per_round.csv",
            },

            "MLPNet-Lite": {
                "fedavg": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_avg_ce-ft3\global_test_per_round.csv",
                "prox001": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.01_ce-ft3\global_test_per_round.csv",
                # "prox005": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.05_ce-ft3\global_test_per_round.csv",
                "prox01": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.1_ce-ft3\global_test_per_round.csv",
                # "prox05": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.5_ce-ft3\global_test_per_round.csv",
            },

            "TabResNet-Lite": {
                "fedavg": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_avg_ce-ft3\global_test_per_round.csv",
                "prox001": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.01_ce-ft3\global_test_per_round.csv",
                # "prox005": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.05_ce-ft3\global_test_per_round.csv",
                "prox01": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.1_ce-ft3\global_test_per_round.csv",
                # "prox05": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.5_ce-ft3\global_test_per_round.csv",
            }
        },

        "50% clients": {

            "LRNet-Lite": {
                "fedavg": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_avg_ce-ft5\global_test_per_round.csv",
                "prox001": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.01_ce-ft5\global_test_per_round.csv",
                # "prox005": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.05_ce-ft5\global_test_per_round.csv",
                "prox01": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.1_ce-ft5\global_test_per_round.csv",
                # "prox05": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.5_ce-ft5\global_test_per_round.csv",
            },

            "MLPNet-Lite": {
                "fedavg": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_avg_ce-ft5\global_test_per_round.csv",
                "prox001": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.01_ce-ft5\global_test_per_round.csv",
                # "prox005": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.05_ce-ft5\global_test_per_round.csv",
                "prox01": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.1_ce-ft5\global_test_per_round.csv",
                # "prox05": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.5_ce-ft5\global_test_per_round.csv",
            },

            "TabResNet-Lite": {
                "fedavg": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_avg_ce-ft5\global_test_per_round.csv",
                "prox001": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.01_ce-ft5\global_test_per_round.csv",
                # "prox005": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.05_ce-ft5\global_test_per_round.csv",
                "prox01": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.1_ce-ft5\global_test_per_round.csv",
                # "prox05": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.5_ce-ft5\global_test_per_round.csv",
            }
        },

        "70% clients": {

            "LRNet-Lite": {
                "fedavg": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_avg_ce-ft7\global_test_per_round.csv",
                "prox001": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.01_ce-ft7\global_test_per_round.csv",
                # "prox005": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.05_ce-ft7\global_test_per_round.csv",
                "prox01": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.1_ce-ft7\global_test_per_round.csv",
                # "prox05": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.5_ce-ft7\global_test_per_round.csv",
            },

            "MLPNet-Lite": {
                "fedavg": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_avg_ce-ft7\global_test_per_round.csv",
                "prox001": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.01_ce-ft7\global_test_per_round.csv",
                # "prox005": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.05_ce-ft7\global_test_per_round.csv",
                "prox01": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.1_ce-ft7\global_test_per_round.csv",
                # "prox05": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.5_ce-ft7\global_test_per_round.csv",
            },

            "TabResNet-Lite": {
                "fedavg": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_avg_ce-ft7\global_test_per_round.csv",
                "prox001": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.01_ce-ft7\global_test_per_round.csv",
                # "prox005": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.05_ce-ft7\global_test_per_round.csv",
                "prox01": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.1_ce-ft7\global_test_per_round.csv",
                # "prox05": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.5_ce-ft7\global_test_per_round.csv",
            }
        },
        "100% clients": {

            "LRNet-Lite": {
                "fedavg": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_avg_ce-ft10\global_test_per_round.csv",
                "prox001": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.01_ce-ft10\global_test_per_round.csv",
                # "prox005": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.05_ce-ft10\global_test_per_round.csv",
                "prox01": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.1_ce-ft10\global_test_per_round.csv",
                # "prox05": "outputs\logistic\exp_logistic_non-iid-0.1_N10_fed_prox-0.5_ce-ft10\global_test_per_round.csv",
            },

            "MLPNet-Lite": {
                "fedavg": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_avg_ce-ft10\global_test_per_round.csv",
                "prox001": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.01_ce-ft10\global_test_per_round.csv",
                # "prox005": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.05_ce-ft10\global_test_per_round.csv",
                "prox01": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.1_ce-ft10\global_test_per_round.csv",
                # "prox05": "outputs\mlp\exp_mlp_non-iid-0.1_N10_fed_prox-0.5_ce-ft10\global_test_per_round.csv",
            },

            "TabResNet-Lite": {
                "fedavg": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_avg_ce-ft10\global_test_per_round.csv",
                "prox001": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.01_ce-ft10\global_test_per_round.csv",
                # "prox005": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.05_ce-ft10\global_test_per_round.csv",
                "prox01": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.1_ce-ft10\global_test_per_round.csv",
                # "prox05": r"outputs\tabresnet\exp_tab-res-net_non-iid-0.1_N10_fed_prox-0.5_ce-ft10\global_test_per_round.csv",
            }
        }

    }

    plot_grid_scenarios_models(
        data_map=data_map,
        metric="global_test_loss",
        # metric="round_train_loss",
        y_label="Global Test Loss",
        y_lim=(0.0, 1.5),
        out_path="charts_outputs/figs/fl_models_non-iid_grid--.png"
    )


if __name__ == "__main__":
    main()