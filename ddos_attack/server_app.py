import time
import json
import csv
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ddos_attack.task import build_model, load_centralized_testloader, test as test_fn

app = ServerApp()


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
    strategy = FedAvg(fraction_train=fraction_train)
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
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
