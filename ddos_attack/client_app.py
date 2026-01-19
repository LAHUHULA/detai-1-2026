"""ddos-attack: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ddos_attack.task import build_model, get_num_features_classes, load_data, get_class_weight_criterion
from ddos_attack.task import test as test_fn
from ddos_attack.task import train as train_fn
import time
from ddos_attack.bench_pi import get_cpu_ram_percent, _try_read_pi_temp_c, get_net_bytes, log_round
from ddos_attack.task import load_data, build_model, train, test, get_num_features_classes_from_local_csv



# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Run config
    model_name = context.run_config.get("model_name", "mlp")
    partition_mode = context.run_config.get("partition_mode", "iid")
    dirichlet_alpha = float(context.run_config.get("dirichlet_alpha", 0.5))
    batch_size = int(context.run_config.get("batch_size", 256))

    # Load model
    partition_id = int(context.node_config["partition-id"])
    mode = context.run_config.get("partition_mode", "iid")
    alpha = float(context.run_config.get("dirichlet_alpha", 0.5))

    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        mode=mode,
        dirichlet_alpha=alpha,
    )

    model = build_model(context.run_config["model_name"], num_features, num_classes)


    # Initialize weights from server
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load partitioned data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
    )

    t0 = time.perf_counter()
    cpu0, ram0 = get_cpu_ram_percent()
    temp0 = _try_read_pi_temp_c()
    tx0, rx0 = get_net_bytes()

    # Train local
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local_epochs"],
        msg.content["config"]["lr"],
        device,
        num_classes=num_classes
    )

        # --- benchmark after ---
    t1 = time.perf_counter()
    cpu1, ram1 = get_cpu_ram_percent()
    temp1 = _try_read_pi_temp_c()
    tx1, rx1 = get_net_bytes()

    log_round({
        "ts": time.time(),
        "client_id": int(context.node_config["partition-id"]),
        "mode": context.run_config.get("partition_mode", "iid"),
        "alpha": float(context.run_config.get("dirichlet_alpha", 0.5)),
        "model": context.run_config.get("model_name", "mlp"),
        "local_epochs": int(context.run_config["local_epochs"]),
        "batch_size": int(context.run_config.get("batch_size", 256)),
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

    # Reply
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Run config
    model_name = context.run_config.get("model_name", "mlp")
    partition_mode = context.run_config.get("partition_mode", "iid")
    dirichlet_alpha = float(context.run_config.get("dirichlet_alpha", 0.5))
    batch_size = int(context.run_config.get("batch_size", 256))

    # Load model
    partition_id = int(context.node_config["partition-id"])
    mode = context.run_config.get("partition_mode", "iid")
    alpha = float(context.run_config.get("dirichlet_alpha", 0.5))

    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        mode=mode,
        dirichlet_alpha=alpha,
    )

    model = build_model(context.run_config["model_name"], num_features, num_classes)


    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load partitioned data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
    )

    criterion = get_class_weight_criterion(device=device)

    # Eval
    eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
    model,
    valloader,
    device,
    num_classes=num_classes,
    criterion=criterion,   # nếu bạn dùng class weights
    )

    metrics = {
        "eval_loss": float(eval_loss),

        "eval_acc": float(acc),

        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),

        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),

        "num-examples": len(valloader.dataset),
    }


    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
