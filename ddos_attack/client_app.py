"""ddos-attack: A Flower / PyTorch ClientApp (real FL on Pi)."""

import time
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ddos_attack.task import (
    build_model,
    load_data,
    get_num_features_classes_from_local_csv,
    train as train_fn,
    test as test_fn,
)

from ddos_attack.bench_pi import (
    get_cpu_ram_percent,
    _try_read_pi_temp_c,
    get_net_bytes,
    log_round,
)

app = ClientApp()


def _get_run_cfg(context: Context):
    """Read run_config with safe defaults."""
    cfg = context.run_config
    return {
        "model_name": cfg.get("model_name", "mlp"),
        "partition_mode": cfg.get("partition_mode", "iid"),
        "dirichlet_alpha": float(cfg.get("dirichlet_alpha", 0.5)),
        "batch_size": int(cfg.get("batch_size", 256)),
        "local_epochs": int(cfg.get("local_epochs", 1)),
    }


def _get_node_cfg(context: Context):
    node_cfg = context.node_config
    return {
        "partition_id": int(node_cfg["partition-id"]),
        "num_partitions": int(node_cfg["num-partitions"]),
    }


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local (client) data."""
    run_cfg = _get_run_cfg(context)
    node_cfg = _get_node_cfg(context)

    partition_id = node_cfg["partition_id"]
    mode = run_cfg["partition_mode"]
    alpha = run_cfg["dirichlet_alpha"]

    # Infer model input/output size from THIS client's CSV
    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        mode=mode,
        dirichlet_alpha=alpha,
    )

    model = build_model(run_cfg["model_name"], num_features, num_classes)

    # Initialize weights from server
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load client data (already MinMaxScaled => no scaling inside task.py)
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=node_cfg["num_partitions"],
        batch_size=run_cfg["batch_size"],
        mode=mode,
        dirichlet_alpha=alpha,
    )

    # --- benchmark before ---
    t0 = time.perf_counter()
    cpu0, ram0 = get_cpu_ram_percent()
    temp0 = _try_read_pi_temp_c()
    tx0, rx0 = get_net_bytes()

    # Train local
    lr = float(msg.content["config"].get("lr", 1e-3))
    train_loss = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=run_cfg["local_epochs"],
        lr=lr,
        device=device,
        num_classes=num_classes,
    )

    # --- benchmark after ---
    t1 = time.perf_counter()
    cpu1, ram1 = get_cpu_ram_percent()
    temp1 = _try_read_pi_temp_c()
    tx1, rx1 = get_net_bytes()

    log_round({
        "ts": time.time(),
        "client_id": partition_id,
        "mode": mode,
        "alpha": alpha,
        "model": run_cfg["model_name"],
        "local_epochs": run_cfg["local_epochs"],
        "batch_size": run_cfg["batch_size"],
        "local_samples": int(len(trainloader.dataset)),
        "train_time_s": float(t1 - t0),
        "cpu_percent_before": cpu0,
        "ram_percent_before": ram0,
        "temp_c_before": temp0,
        "cpu_percent_after": cpu1,
        "ram_percent_after": ram1,
        "temp_c_after": temp1,
        "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
        "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and rx1 >= 0) else None,
    })

    # Reply: weights + metrics
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord({
        "train_loss": float(train_loss),
        "num-examples": int(len(trainloader.dataset)),
    })
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local validation data."""
    run_cfg = _get_run_cfg(context)
    node_cfg = _get_node_cfg(context)

    partition_id = node_cfg["partition_id"]
    mode = run_cfg["partition_mode"]
    alpha = run_cfg["dirichlet_alpha"]

    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        mode=mode,
        dirichlet_alpha=alpha,
    )

    model = build_model(run_cfg["model_name"], num_features, num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use client's local val split
    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=node_cfg["num_partitions"],
        batch_size=run_cfg["batch_size"],
        mode=mode,
        dirichlet_alpha=alpha,
    )

    # Eval (criterion default CrossEntropyLoss inside test_fn if None)
    eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
        net=model,
        testloader=valloader,
        device=device,
        num_classes=num_classes,
        criterion=None,
    )

    metric_record = MetricRecord({
        "eval_loss": float(eval_loss),
        "eval_acc": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "num-examples": int(len(valloader.dataset)),
    })

    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
