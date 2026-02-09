import time
import json
import csv
from pathlib import Path
import os
from typing import Any, Dict, Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx

from ddos_attack.task import build_model, load_centralized_testloader, test as test_fn

app = ServerApp()

# ---- ADD: FedAvg subclass to run centralized test each round ----
class ServerTestFedProx(FedProx):
    def __init__(
        self,
        *args,
        model_name: str,
        num_features: int,
        num_classes: int,
        batch_size: int,
        test_csv_path: str,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)

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
        # Build model, load current aggregated weights
        model = build_model(self.model_name, self.num_features, self.num_classes).to(self.device)
        state_dict = arrays_record.to_torch_state_dict()
        model.load_state_dict(state_dict, strict=True)
        # ---- SAVE MODEL CHECKPOINT PER ROUND ----
        ckpt_path = self.out_dir / f"{self.model_name}_r{int(server_round)}.pt"
        # Save state_dict (portable)
        torch.save(state_dict, ckpt_path)

        # Centralized test on test_final.csv
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
        self._append_csv(row)

    # Hook after aggregation each round
    def aggregate_train(self, server_round, results, *args, **kwargs):
        agg = super().aggregate_train(server_round, results, *args, **kwargs)

        # agg is usually (arrays, metrics) in Flower serverapp strategy
        if isinstance(agg, tuple) and len(agg) == 2:
            arrays_record, _ = agg
        else:
            arrays_record = agg

        # Only test if we actually have a model
        if arrays_record is not None:
            self._server_test(server_round=int(server_round), arrays_record=arrays_record)

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
    num_features = int(cfg.get("num-features", 40))
    num_classes = int(cfg.get("num-classes", 13))

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
        fraction_train=fraction_train,
        fraction_evaluate=fraction_train,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        proximal_mu=0.00,
        model_name=model_name,
        num_features=num_features,
        num_classes=num_classes,
        batch_size=batch_size,
        test_csv_path=server_test_csv,
        output_dir=str(out_dir),
    )
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "proximal_mu": 0.00,
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
    avg_round_time = total_time / max(num_rounds, 1)
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
        "model_size_kib": float(model_bytes / 1024.0),
        "est_comm_per_round_mib": est_comm_per_round_mib,
        **centralized,
        "final_model_path": str(model_path),
    }
    _save_json(out_dir / "run_summary.json", run_summary)

    # Append global summary CSV
    summary_path = out_root / "results_summary.csv"
    header = [
        "exp_id", "model", "partition_mode", "dirichlet_alpha",
        "num_clients", "num_rounds", "fraction_train", "local_epochs",
        "batch_size", "lr", "num_features", "num_classes",
        "total_time_s", "avg_round_time_s", "model_size_kib", "est_comm_per_round_mib",
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
