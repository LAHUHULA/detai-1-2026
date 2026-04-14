import time
import json
import csv
from pathlib import Path
import random
from typing import Any, Dict, Optional

import numpy as np                         # [ADD]
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # [ADD MetricRecord]
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx, FedXgbBagging  # [ADD]

from ddos_attack.task import (
    build_model,
    load_centralized_testloader,
    load_centralized_test_numpy,          # [ADD]
    build_xgb_params_from_cfg,            # [ADD]
    xgb_evaluate_bytes,                   # [ADD]
    test as test_fn,
)
# ============================================================
# [ADD] Lazy import for XGBoost
# ============================================================

def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "XGBoost is required but not installed in the Flower runtime environment.\n"
            "Fix:\n"
            "  1) Add xgboost to [project].dependencies in pyproject.toml\n"
            "  2) Bump project.version\n"
            "  3) Re-run `flwr run ...` so Flower rebuilds the app bundle\n"
            "  4) Or install manually on the server machine: pip install xgboost\n"
        ) from e

app = ServerApp()

# ============================================================
# [ADD] XGBoost server-side evaluation helper
# ============================================================

def _server_side_xgb_eval(server_round: int, arrays: ArrayRecord, cfg: dict, num_classes: int):
    # [ADD] guard empty arrays
    if "0" not in arrays:
        print(f"[WARN] Round {server_round}: no XGBoost arrays found, skip server eval.")
        return None

    model_bytes = bytearray(arrays["0"].numpy().tobytes())
    if len(model_bytes) == 0:
        print(f"[WARN] Round {server_round}: empty XGBoost model, skip server eval.")
        return None

    test_csv = str(cfg.get("server-test-csv", "")).strip()
    if not test_csv:
        print(f"[WARN] Round {server_round}: no server-test-csv configured, skip server eval.")
        return None

    try:
        X_test, y_test = load_centralized_test_numpy(test_csv)
        loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = xgb_evaluate_bytes(
            model_bytes=model_bytes,
            X=X_test,
            y=y_test,
            cfg=cfg,
            num_classes=num_classes,
        )

        return MetricRecord({
            "global_test_loss": float(loss),
            "global_test_acc": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_w),
            "recall_weighted": float(r_w),
            "f1_weighted": float(f1_w),
        })
    except Exception as e:
        print(f"[WARN] Round {server_round}: server-side XGBoost eval skipped due to error: {e}")
        return None

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
        # ============================================================
        # [ADD] SAFE BUILD PARTITION MAP FROM VALID REPLIES ONLY
        # ============================================================
        valid_results = []

        for reply in results:
            node_id = reply.metadata.src_node_id

            # Skip replies without content
            if not reply.has_content():
                if reply.has_error():
                    try:
                        print(f"[WARN] Round {server_round}: node {node_id} returned error: {reply.error}")
                    except Exception:
                        print(f"[WARN] Round {server_round}: node {node_id} returned an error reply.")
                else:
                    print(f"[WARN] Round {server_round}: node {node_id} returned empty reply.")
                continue

            valid_results.append(reply)

            # --------------------------------------------------------
            # [KEEP] Read partition-id for mapping node_id <-> partition_id
            # --------------------------------------------------------
            try:
                metric_record = reply.content["metrics"]
                metrics = dict(metric_record)

                if "partition-id" in metrics:
                    pid = int(metrics["partition-id"])
                    self.partition_to_node[pid] = node_id
            except Exception as e:
                print(f"[WARN] Round {server_round}: failed to read partition-id from node {node_id}: {e}")

        print("UPDATED PARTITION → NODE MAP:")
        print(self.partition_to_node)

        # If no valid replies, fail with clear message
        if len(valid_results) == 0:
            raise RuntimeError(
                f"No valid client replies with content in round {server_round}. "
                "Check client logs for the original exception."
            )

        # ============================================================
        # [KEEP] NORMAL AGGREGATION USING ONLY VALID REPLIES
        # ============================================================
        agg = super().aggregate_train(server_round, valid_results, *args, **kwargs)

        # ---- EXTRACT AGGREGATED TRAIN LOSS
        if isinstance(agg, tuple) and len(agg) == 2:
            arrays_record, aggregated_metrics = agg
        else:
            arrays_record = agg
            aggregated_metrics = None

        if aggregated_metrics is not None:
            metrics_dict = dict(aggregated_metrics)
            if "train_loss" in metrics_dict:
                self.round_train_loss[int(server_round)] = float(metrics_dict["train_loss"])

            # [ADD] partition-id aggregated value is meaningless, ignore it
            if "partition-id" in metrics_dict:
                print(
                    f"[INFO] Round {server_round}: aggregated 'partition-id'={metrics_dict['partition-id']} "
                    f"(ignored, only per-client partition-id is used for mapping)"
                )

        # ============================================================
        # [KEEP] Round 1 only used to build mapping
        # ============================================================
        if int(server_round) == 1:
            print("⚠ Round 1 used only for mapping. Resetting global model.")

            arrays_record = self.initial_arrays

            if hasattr(self, "round_train_loss"):
                self.round_train_loss.pop(1, None)

            if hasattr(self, "round_wall_time"):
                self.round_wall_time.pop(1, None)

            # DO NOT run server-side test on round 1
            return arrays_record, {}

        # ============================================================
        # [KEEP] Normal behavior from round >= 2
        # ============================================================
        if arrays_record is not None:
            self._server_test(server_round, arrays_record)

        return agg

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

    num_rounds = int(cfg["num-server-rounds"])
    fraction_train = float(cfg["fraction-train"])
    fraction_evaluate = float(cfg.get("fraction-evaluate", 1.0))
    lr = float(cfg.get("lr", 1e-3))
    local_epochs = int(cfg["local-epochs"])
    batch_size = int(cfg.get("batch-size", 256))

    model_name = str(cfg.get("model-name", "mlp")).lower().strip()
    strategy_name = str(cfg.get("strategy-name", "fedprox")).lower().strip()

    num_clients = int(cfg.get("num-clients", 7))
    num_features = int(cfg.get("num-features", 39))
    num_classes = int(cfg.get("num-classes", 13))

    partition_mode = str(cfg.get("partition-mode", "iid"))
    dirichlet_alpha = float(cfg.get("dirichlet-alpha", 0.5))

    do_centralized_test = bool(cfg.get("do-centralized-test", True))
    server_test_csv = str(cfg.get("server-test-csv", ""))

    _save_json(out_dir / "run_config.json", dict(cfg))

    # ============================================================
    # [ADD] XGBoost + FedXgbBagging branch
    # ============================================================
    if model_name == "xgboost":
        if strategy_name != "fedxgbbagging":
            raise ValueError("For model-name='xgboost', set strategy-name='fedxgbbagging'")

        initial_arrays = ArrayRecord([np.frombuffer(b"", dtype=np.uint8)])

        strategy = FedXgbBagging(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=num_clients,
            min_evaluate_nodes=num_clients,
            min_available_nodes=num_clients,
        )

        t0 = time.perf_counter()
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=ConfigRecord({}),
            num_rounds=num_rounds,
            evaluate_fn=(
                (lambda server_round, arrays: _server_side_xgb_eval(server_round, arrays, cfg, num_classes))
                if do_centralized_test else None
            ),
        )
        t1 = time.perf_counter()

        final_bytes = bytearray(result.arrays["0"].numpy().tobytes())
        model_path = out_dir / "final_model_xgboost.json"

        xgb = _require_xgboost()
        bst = xgb.Booster(params=build_xgb_params_from_cfg(cfg, num_classes))
        bst.load_model(final_bytes)
        bst.save_model(str(model_path))

        total_time = float(t1 - t0)
        avg_round_time = total_time / max(num_rounds, 1)
        model_bytes = len(final_bytes)
        clients_per_round = max(1, int(round(fraction_train * num_clients)))
        est_comm_per_round_mib = float((model_bytes * 2 * clients_per_round) / (1024.0 ** 2))

        centralized = {}
        if do_centralized_test:
            metric_record = _server_side_xgb_eval(num_rounds, result.arrays, cfg, num_classes)
            if metric_record is not None:
                centralized = dict(metric_record)

        run_summary = {
            "exp_id": exp_id,
            "model": model_name,
            "strategy": strategy_name,
            "partition_mode": partition_mode,
            "dirichlet_alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "fraction_evaluate": fraction_evaluate,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": None,
            "num_features": num_features,
            "num_classes": num_classes,
            "total_time_s": total_time,
            "avg_round_time_s": float(avg_round_time),
            "server_test_time_total_s": None,
            "avg_round_server_test_time_s": None,
            "model_size_kib": float(model_bytes / 1024.0),
            "est_comm_per_round_mib": est_comm_per_round_mib,
            **centralized,
            "final_model_path": str(model_path),
        }

        _save_json(out_dir / "run_summary.json", run_summary)

        summary_path = out_root / "results_summary.csv"
        header = [
            "exp_id", "model", "strategy", "partition_mode", "dirichlet_alpha",
            "num_clients", "num_rounds", "fraction_train", "fraction_evaluate", "local_epochs",
            "batch_size", "lr", "num_features", "num_classes",
            "total_time_s", "avg_round_time_s",
            "server_test_time_total_s", "avg_round_server_test_time_s",
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
        return

    # ============================================================
    # Existing PyTorch / FedProx branch
    # ============================================================
    sampling_plan_path = out_dir / "sampling_plan.json"

    sampling_plan = generate_sampling_plan(
        num_clients=num_clients,
        fraction_train=fraction_train,
        num_rounds=num_rounds,
        out_path=sampling_plan_path,
    )

    global_model = build_model(model_name, num_features, num_classes)
    initial_arrays = ArrayRecord(global_model.state_dict())

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

    state_dict = result.arrays.to_torch_state_dict()
    model_path = out_dir / f"final_model_{model_name}.pt"
    torch.save(state_dict, str(model_path))

    total_time = float(t1 - t0)
    effective_rounds = max(num_rounds - 1, 1)
    avg_round_time = total_time / effective_rounds
    server_test_time_total_s = float(getattr(strategy, "server_test_time_total_s", 0.0))
    avg_round_server_test_time_s = server_test_time_total_s / max(num_rounds, 1)
    model_bytes = sum(v.numel() * v.element_size() for v in state_dict.values() if torch.is_tensor(v))
    clients_per_round = max(1, int(round(fraction_train * num_clients)))
    est_comm_per_round_mib = float((model_bytes * 2 * clients_per_round) / (1024.0 ** 2))

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

    run_summary = {
        "exp_id": exp_id,
        "model": model_name,
        "strategy": strategy_name,
        "partition_mode": partition_mode,
        "dirichlet_alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_features": num_features,
        "num_classes": num_classes,
        "total_time_s": total_time,
        "avg_round_time_s": float(avg_round_time),
        "server_test_time_total_s": float(server_test_time_total_s),
        "avg_round_server_test_time_s": float(avg_round_server_test_time_s),
        "model_size_kib": float(model_bytes / 1024.0),
        "est_comm_per_round_mib": est_comm_per_round_mib,
        **centralized,
        "final_model_path": str(model_path),
    }

    run_summary["round_train_loss"] = strategy.round_train_loss
    run_summary["round_wall_time_s"] = strategy.round_wall_time
    _save_json(out_dir / "run_summary.json", run_summary)

    summary_path = out_root / "results_summary.csv"
    header = [
        "exp_id", "model", "strategy", "partition_mode", "dirichlet_alpha",
        "num_clients", "num_rounds", "fraction_train", "fraction_evaluate", "local_epochs",
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