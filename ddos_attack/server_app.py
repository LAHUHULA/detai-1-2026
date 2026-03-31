import time
import json
import csv
from pathlib import Path
import os
import random
from typing import Any, Dict, Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx

from ddos_attack.task import build_model, load_centralized_testloader, test as test_fn

app = ServerApp()

# ---- ADD: FedAvg subclass to run centralized test each round ----
# ---- ADD: FedAvg subclass to run centralized test each round ----
class ServerTestFedProx(FedProx):
    def __init__(
        self,
        *args,
        sampling_plan: dict,
        model_name: str,
        num_features: int,
        num_classes: int,
        batch_size: int,
        test_csv_path: str,
        output_dir: Optional[str] = None,
        initial_arrays = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_arrays = initial_arrays
        self.sampling_plan = sampling_plan
        self.model_name = model_name
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)

        # ==== [ADD] time accumulators for "FL core time" estimation
        self.server_test_time_total_s: float = 0.0
        self.server_test_time_by_round: dict[int, float] = {}

        # ==== [ADD] round-level train metrics
        self.round_train_loss: dict[int, float] = {}
        self.round_fl_core_time: dict[int, float] = {}

        # ==== [FIX] round wall-clock timing
        self._round_start_time: dict[int, float] = {}

        # ==== [ADD] precise round timing
        self.round_wall_time: dict[int, float] = {}

        # output dir (prefer env OUTPUT_DIR -> outputs/<exp_id>)
        if output_dir and str(output_dir).strip():
            self.out_dir = Path(output_dir).expanduser().resolve()
        else:
            self.out_dir = (Path(__file__).resolve().parent.parent / "outputs" / "default").resolve()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "global_test_per_round.csv"
        self._header_written = self.csv_path.exists()

        # Prepare once
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_csv_path = str(test_csv_path)
        self.testloader = load_centralized_testloader(self.test_csv_path, batch_size=self.batch_size)

        self.partition_to_node = {}


    def configure_train(self, server_round, arrays, config, grid):

        self._round_start_time[int(server_round)] = time.perf_counter()

        selected_partitions = set(
            int(x) for x in self.sampling_plan.get(str(server_round), [])
        )

        print(f"\n========== ROUND {server_round} ==========")
        print("PLAN:", selected_partitions)
        print("CURRENT MAP:", self.partition_to_node)

        # Lấy messages gốc
        messages = super().configure_train(server_round, arrays, config, grid)

        # Nếu mapping chưa đủ (round 1), cho tất cả train
        if len(self.partition_to_node) == 0:
            print("⚠ Mapping not ready yet. Let all nodes train.")
            return messages

        selected_node_ids = {
            self.partition_to_node[p]
            for p in selected_partitions
            if p in self.partition_to_node
        }

        print("SELECTED NODE IDS:", selected_node_ids)
        print("================================")

        filtered = [
            msg for msg in messages
            if msg.metadata.dst_node_id in selected_node_ids
        ]

        return filtered

    def _append_csv(self, row: Dict[str, Any]) -> None:
        # write header once
        if not self._header_written:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
            self._header_written = True
        else:
            # keep existing header stable
            with open(self.csv_path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            for k in header:
                row.setdefault(k, None)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writerow(row)

    def _server_test(self, server_round: int, arrays_record) -> None:
        # ==== [FIX] compute round wall time here to guarantee availability
        start_t = self._round_start_time.get(int(server_round))
        if start_t is not None:
            self.round_wall_time[int(server_round)] = float(time.perf_counter() - start_t)
        # ==== [ADD] measure server-side test overhead per round
        _t0 = time.perf_counter()

        # Build model, load current aggregated weights
        model = build_model(self.model_name, self.num_features, self.num_classes).to(self.device)
        state_dict = arrays_record.to_torch_state_dict()
        model.load_state_dict(state_dict, strict=True)

        # ---- SAVE MODEL CHECKPOINT PER ROUND ----
        ckpt_path = self.out_dir / f"{self.model_name}_r{int(server_round)}.pt"
        torch.save(state_dict, ckpt_path)

        # Centralized test on test CSV
        loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
            model,
            self.testloader,
            self.device,
            num_classes=self.num_classes,
            criterion=None,
        )

        row = {
            "round": int(server_round),
            "global_test_loss": float(loss),
            "global_test_acc": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_w),
            "recall_weighted": float(r_w),
            "f1_weighted": float(f1_w),
            "checkpoint_path": str(ckpt_path),
        }

        # ==== [ADD] dt_no_csv: inference + ckpt save (no csv yet)
        _t1 = time.perf_counter()
        dt_no_csv = float(_t1 - _t0)

        # ==== attach optional round info
        if int(server_round) in self.round_train_loss:
            row["round_train_loss"] = self.round_train_loss[int(server_round)]

        if int(server_round) in self.round_wall_time:
            row["round_wall_time_s"] = self.round_wall_time[int(server_round)]

        # ==== measure CSV write time
        _t2 = time.perf_counter()
        row["server_test_time_s"] = dt_no_csv
        self._append_csv(row)
        _t3 = time.perf_counter()

        dt_csv = float(_t3 - _t2)

        # ==== total server-test time (inference + csv)
        dt_total = dt_no_csv + dt_csv

        # now safe to log it
        row["server_test_time_s"] = dt_total

        # accumulate
        self.server_test_time_total_s += dt_total
        self.server_test_time_by_round[int(server_round)] = dt_total


    def aggregate_train(self, server_round, results, *args, **kwargs):

        # ---- BUILD PARTITION MAP (giữ nguyên)
        for reply in results:
            node_id = reply.metadata.src_node_id
            metric_record = reply.content["metrics"]
            metrics = dict(metric_record)

            if "partition-id" in metrics:
                pid = int(metrics["partition-id"])
                self.partition_to_node[pid] = node_id

        print("UPDATED PARTITION → NODE MAP:")
        print(self.partition_to_node)

        # ---- NORMAL AGGREGATION
        agg = super().aggregate_train(server_round, results, *args, **kwargs)

        # ---- EXTRACT AGGREGATED TRAIN LOSS
        if isinstance(agg, tuple) and len(agg) == 2:
            arrays_record, aggregated_metrics = agg
        else:
            arrays_record = agg
            aggregated_metrics = None

        if aggregated_metrics is not None:
            metrics_dict = dict(aggregated_metrics)
            if "train_loss" in metrics_dict:
                self.round_train_loss[int(server_round)] = float(
                    metrics_dict["train_loss"]
                )

        # Round 1 only used to build mapping
        if int(server_round) == 1:
            print("⚠ Round 1 used only for mapping. Resetting global model.")

            # Reset global model to initial state
            arrays_record = self.initial_arrays

            # Remove round 1 metrics
            if hasattr(self, "round_train_loss"):
                self.round_train_loss.pop(1, None)

            if hasattr(self, "round_wall_time"):
                self.round_wall_time.pop(1, None)

            # DO NOT run server-side test
            return arrays_record, {}

        # Normal behavior from round >= 2
        if arrays_record is not None:
            self._server_test(server_round, arrays_record)

        return agg
    # Hook after aggregation each round
    # def aggregate_train(self, server_round, results, *args, **kwargs):

    #     # ---- normal aggregation
    #     agg = super().aggregate_train(server_round, results, *args, **kwargs)

    #     # ==== [ROUND END MARK + COMPUTE WALL TIME]
    #     start_t = self._round_start_time.get(int(server_round))

    #     if start_t is not None:
    #         round_time = float(time.perf_counter() - start_t)
    #         self.round_wall_time[int(server_round)] = round_time

    #     # ---- extract arrays + metrics safely
    #     if isinstance(agg, tuple) and len(agg) == 2:
    #         arrays_record, metrics = agg
    #     else:
    #         arrays_record = agg
    #         metrics = None

    #     if metrics is not None:
    #         try:
    #             metrics_dict = dict(metrics)
    #             train_loss = metrics_dict.get("train_loss")
    #             if train_loss is not None:
    #                 self.round_train_loss[int(server_round)] = float(train_loss)
    #         except Exception:
    #             pass

    #     if arrays_record is not None:
    #         self._server_test(server_round=int(server_round), arrays_record=arrays_record)

    #     return agg

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_csv_row(path: Path, header: list[str], row: dict) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def generate_sampling_plan(
    num_clients: int,
    fraction_train: float,
    num_rounds: int,
    out_path: Path,
    seed: int = 84,
):
    random.seed(seed)
    clients = list(range(num_clients))
    plan = {}

    k = max(1, int(num_clients * fraction_train))

    for r in range(1, num_rounds + 1):
        selected = random.sample(clients, k)
        plan[str(r)] = selected

    out_path.write_text(json.dumps(plan, indent=2))
    return plan

@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    exp_id = str(cfg.get("exp-id", "exp")).strip()
    out_root = Path(str(cfg.get("output-dir", "outputs")))
    out_dir = _ensure_dir(out_root / exp_id)

    # Read config (kebab-case from pyproject.toml)
    num_rounds = int(cfg["num-server-rounds"])
    fraction_train = float(cfg["fraction-train"])
    lr = float(cfg["lr"])
    local_epochs = int(cfg["local-epochs"])
    batch_size = int(cfg.get("batch-size", 256))

    model_name = str(cfg.get("model-name", "mlp"))
    num_clients = int(cfg.get("num-clients", 7))
    num_features = int(cfg.get("num-features", 39))
    num_classes = int(cfg.get("num-classes", 13))

    sampling_plan_path = out_dir / "sampling_plan.json"

    sampling_plan = generate_sampling_plan(
        num_clients=num_clients,
        fraction_train=fraction_train,
        num_rounds=num_rounds,
        out_path=sampling_plan_path,
    )

    partition_mode = str(cfg.get("partition-mode", "iid"))
    dirichlet_alpha = float(cfg.get("dirichlet-alpha", 0.5))

    do_centralized_test = bool(cfg.get("do-centralized-test", True))
    server_test_csv = str(cfg.get("server-test-csv", ""))

    # Save run config snapshot
    _save_json(out_dir / "run_config.json", dict(cfg))

    # Build global model
    global_model = build_model(model_name, num_features, num_classes)
    initial_arrays = ArrayRecord(global_model.state_dict())

    # Start FL
    strategy = ServerTestFedProx(
        sampling_plan=sampling_plan,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=num_clients,
        min_evaluate_nodes=1,
        min_available_nodes=num_clients,
        proximal_mu=0.01,
        model_name=model_name,
        num_features=num_features,
        num_classes=num_classes,
        batch_size=batch_size,
        test_csv_path=server_test_csv,
        output_dir=str(out_dir),
        initial_arrays=initial_arrays,
    )
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "proximal_mu": 0.01,
        }),
        num_rounds=num_rounds,
    )
    t1 = time.perf_counter()

    # Save final model
    state_dict = result.arrays.to_torch_state_dict()
    model_path = out_dir / f"final_model_{model_name}.pt"
    torch.save(state_dict, str(model_path))

    # Estimate comm/time
    total_time = float(t1 - t0)
    effective_rounds = max(num_rounds - 1, 1)
    avg_round_time = total_time / effective_rounds
    # ==== [ADD] estimate "FL core time" by subtracting server_test time
    server_test_time_total_s = float(getattr(strategy, "server_test_time_total_s", 0.0))
    # fl_core_time_s = max(0.0, float(total_time - server_test_time_total_s))
    # avg_round_fl_core_time_s = fl_core_time_s / max(num_rounds, 1)
    avg_round_server_test_time_s = server_test_time_total_s / max(num_rounds, 1)
    model_bytes = sum(v.numel() * v.element_size() for v in state_dict.values() if torch.is_tensor(v))
    clients_per_round = max(1, int(round(fraction_train * num_clients)))
    est_comm_per_round_mib = float((model_bytes * 2 * clients_per_round) / (1024.0 ** 2))

    # Centralized test
    centralized = {}
    if do_centralized_test:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = build_model(model_name, num_features, num_classes).to(device)
        model.load_state_dict(state_dict)

        testloader = load_centralized_testloader(server_test_csv, batch_size=batch_size)
        loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
            model, testloader, device, num_classes=num_classes, criterion=None
        )
        centralized = {
            "global_test_loss": float(loss),
            "global_test_acc": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_w),
            "recall_weighted": float(r_w),
            "f1_weighted": float(f1_w),
        }

    # Save per-run summary json
    run_summary = {
        "exp_id": exp_id,
        "model": model_name,
        "partition_mode": partition_mode,
        "dirichlet_alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_features": num_features,
        "num_classes": num_classes,
        "total_time_s": total_time,
        "avg_round_time_s": float(avg_round_time),
        # ==== [ADD] time breakdown
        "server_test_time_total_s": float(server_test_time_total_s),
        # "fl_core_time_s": float(fl_core_time_s),
        # "avg_round_fl_core_time_s": float(avg_round_fl_core_time_s),
        "avg_round_server_test_time_s": float(avg_round_server_test_time_s),
        "model_size_kib": float(model_bytes / 1024.0),
        "est_comm_per_round_mib": est_comm_per_round_mib,
        **centralized,
        "final_model_path": str(model_path),
    }
        # ==== [ADD] round-level logs
    run_summary["round_train_loss"] = strategy.round_train_loss
    run_summary["round_wall_time_s"] = strategy.round_wall_time
    _save_json(out_dir / "run_summary.json", run_summary)

    # Append global summary CSV
    summary_path = out_root / "results_summary.csv"
    # "fl_core_time_s", "avg_round_fl_core_time_s"
    header = [
        "exp_id", "model", "partition_mode", "dirichlet_alpha",
        "num_clients", "num_rounds", "fraction_train", "local_epochs",
        "batch_size", "lr", "num_features", "num_classes",
        "total_time_s", "avg_round_time_s", "server_test_time_total_s",
        "avg_round_server_test_time_s", 
        "model_size_kib", "est_comm_per_round_mib",
        "global_test_loss", "global_test_acc",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "final_model_path",
    ]
    for k in header:
        run_summary.setdefault(k, None)
    _append_csv_row(summary_path, header, run_summary)

    print(f"\n[OK] Saved: {out_dir}")
    print(f"[OK] Updated: {summary_path}")
