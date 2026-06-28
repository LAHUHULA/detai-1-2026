import time
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ddos_attack.task import (
    build_model,
    load_data,
    load_data_numpy,
    xgb_train_one_client,
    xgb_train_one_client_cyclic,      # [ADD]
    xgb_evaluate_bytes,
    build_xgbllr_meta_model,          # [ADD]
    xgb_models_from_json,             # [ADD]
    build_xgbllr_features_from_models, # [ADD]
    train_xgbllr_meta,                # [ADD]
    test_xgbllr_meta,                 # [ADD]
    compute_class_weights_from_labels, # [ADD]
    lgb_train_one_client,
    lgb_evaluate_bytes,
    train as train_fn,
    test as test_fn,
    get_num_features_classes_from_local_csv,
)
# [ADD]
import numpy as np
from ddos_attack.bench_pi import get_cpu_ram_percent, _try_read_pi_temp_c, get_net_bytes, log_round
from ddos_attack.bench_pi import ResourceMonitor

app = ClientApp()


def _k(cfg: dict, key: str, default=None):
    # kebab-case keys from pyproject.toml
    return cfg.get(key, default)


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config

    model_name = str(_k(cfg, "model-name", "mlp")).lower().strip()
    partition_mode = str(_k(cfg, "partition-mode", "iid"))
    dirichlet_alpha = float(_k(cfg, "dirichlet-alpha", 0.5))
    batch_size = int(_k(cfg, "batch-size", 256))
    data_root = str(_k(cfg, "data-root", ""))
    local_epochs = int(_k(cfg, "local-epochs", 1))

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        num_partitions=num_partitions,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
    )

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    t0 = time.perf_counter()
    cpu0, ram0 = get_cpu_ram_percent()
    temp0 = _try_read_pi_temp_c()
    tx0, rx0 = get_net_bytes()

    # ============================================================
    # [ADD] LightGBM branch
    # ============================================================

    if model_name == "lightgbm":
        strategy_name = str(_k(cfg, "strategy-name", "fedxgbbagging")).lower().strip()

        X_train, y_train, _, _ = load_data_numpy(
            partition_id=partition_id,
            num_partitions=num_partitions,
            mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            data_root=data_root,
        )

        global_model_bytes = b""
        if "arrays" in msg.content and "0" in msg.content["arrays"]:
            global_model_bytes = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # ------------------------------------------------------------
        # [ADD] LightGBM branch
        # ------------------------------------------------------------
        if strategy_name == "fedxgbbagging":
            local_model_bytes = lgb_train_one_client(
                global_model_bytes=global_model_bytes,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
                num_classes=num_classes,
                num_local_round=local_epochs,
            )
        elif strategy_name == "fedxgbcyclic":
            local_model_bytes = xgb_train_one_client_cyclic(
                global_model_bytes=global_model_bytes,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
                num_classes=num_classes,
                num_local_round=local_epochs,
            )
        elif strategy_name == "fedlgb":
            local_model_bytes = lgb_train_one_client(  # [ADD] LightGBM
                global_model_bytes=global_model_bytes,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
                num_classes=num_classes,
                num_local_round=local_epochs,
            )
        else:
            raise ValueError(
                f"Unsupported strategy-name='{strategy_name}' for model-name='{model_name}'. "
                f"Use 'fedxgbbagging', 'fedxgbcyclic', or 'fedlgb'."
            )

        t1 = time.perf_counter()
        monitor.stop()
        resource_stats = monitor.summary()

        cpu1, ram1 = get_cpu_ram_percent()
        temp1 = _try_read_pi_temp_c()
        tx1, rx1 = get_net_bytes()

        log_round({
            "exp_id": str(_k(cfg, "exp-id", "")),
            "client_id": partition_id,
            "model": model_name,
            "mode": partition_mode,
            "alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": None,
            "local_samples": int(len(y_train)),
            "train_time_s": float(t1 - t0),
            **resource_stats,
            "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
            "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and rx1 >= 0) else None,
            "param_count": None,
            "model_size_kib": float(len(local_model_bytes) / 1024.0),
        })

        model_np = np.frombuffer(local_model_bytes, dtype=np.uint8)
        model_record = ArrayRecord([model_np])

        metrics = {
            "train_loss": 0.0,
            "num-examples": int(len(y_train)),
            "partition-id": int(partition_id),
        }
        content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    # ============================================================
    # [ADD] XGBoost branch
    # ============================================================
    if model_name == "xgboost":
        strategy_name = str(_k(cfg, "strategy-name", "fedxgbbagging")).lower().strip()

        X_train, y_train, _, _ = load_data_numpy(
            partition_id=partition_id,
            num_partitions=num_partitions,
            mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            data_root=data_root,
        )

        global_model_bytes = b""
        if "arrays" in msg.content and "0" in msg.content["arrays"]:
            global_model_bytes = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # ============================================================
        # [ADD] FedXGBllr branch
        # ============================================================
        if strategy_name == "fedxgbllr":
            phase = str(msg.content["config"].get("phase", "bootstrap_xgbllr"))

            # -------------------- ROUND 1: bootstrap + mapping --------------------
            if phase == "bootstrap_xgbllr":
                local_xgb_rounds = int(msg.content["config"].get("local_xgb_rounds", local_epochs))

                # Use cyclic-style helper with empty global model to return FULL local ensemble
                local_model_bytes = xgb_train_one_client_cyclic(
                    global_model_bytes=b"",
                    X_train=X_train,
                    y_train=y_train,
                    cfg=cfg,
                    num_classes=num_classes,
                    num_local_round=local_xgb_rounds,
                )

                t1 = time.perf_counter()
                monitor.stop()
                resource_stats = monitor.summary()

                cpu1, ram1 = get_cpu_ram_percent()
                temp1 = _try_read_pi_temp_c()
                tx1, rx1 = get_net_bytes()

                log_round({
                    "exp_id": str(_k(cfg, "exp-id", "")),
                    "client_id": partition_id,
                    "model": "xgboost-fllr-bootstrap",
                    "mode": partition_mode,
                    "alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
                    "local_epochs": local_xgb_rounds,
                    "batch_size": batch_size,
                    "lr": None,
                    "local_samples": int(len(y_train)),
                    "train_time_s": float(t1 - t0),
                    **resource_stats,
                    "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
                    "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and rx1 >= 0) else None,
                    "param_count": None,
                    "model_size_kib": float(len(local_model_bytes) / 1024.0),
                })

                model_np = np.frombuffer(local_model_bytes, dtype=np.uint8)
                model_record = ArrayRecord([model_np])

                metrics = {
                    "train_loss": 0.0,
                    "num-examples": int(len(y_train)),
                    "partition-id": int(partition_id),
                }
                content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
                return Message(content=content, reply_to=msg)

            # -------------------- ROUND >=2: train meta learner --------------------
            elif phase == "train_meta_xgbllr":
                xgb_models_json = str(msg.content["config"]["xgb_models_json"])
                meta_lr = float(msg.content["config"].get("lr", 1e-3))

                xgb_model_list = xgb_models_from_json(xgb_models_json)
                meta_features = build_xgbllr_features_from_models(
                    xgb_model_list, X_train, cfg, num_classes
                )

                class_weights = compute_class_weights_from_labels(y_train, num_classes)

                meta_model = build_xgbllr_meta_model(
                    num_models=len(xgb_model_list),
                    num_classes=num_classes,
                )
                meta_model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                meta_model.to(device)

                train_loss = train_xgbllr_meta(
                    model=meta_model,
                    features_np=meta_features,
                    labels_np=y_train,
                    epochs=local_epochs,
                    lr=meta_lr,
                    batch_size=batch_size,
                    device=device,
                    class_weights=class_weights,
                )

                t1 = time.perf_counter()
                monitor.stop()
                resource_stats = monitor.summary()

                cpu1, ram1 = get_cpu_ram_percent()
                temp1 = _try_read_pi_temp_c()
                tx1, rx1 = get_net_bytes()

                param_count = sum(p.numel() for p in meta_model.parameters())
                model_bytes = sum(p.numel() * p.element_size() for p in meta_model.parameters())

                log_round({
                    "exp_id": str(_k(cfg, "exp-id", "")),
                    "client_id": partition_id,
                    "model": "fedxgbllr-meta",
                    "mode": partition_mode,
                    "alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
                    "local_epochs": local_epochs,
                    "batch_size": batch_size,
                    "lr": float(meta_lr),
                    "local_samples": int(len(y_train)),
                    "train_time_s": float(t1 - t0),
                    **resource_stats,
                    "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
                    "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and rx1 >= 0) else None,
                    "param_count": int(param_count),
                    "model_size_kib": float(model_bytes / 1024.0),
                })

                model_record = ArrayRecord(meta_model.state_dict())
                metrics = {
                    "train_loss": float(train_loss),
                    "num-examples": int(len(y_train)),
                    "partition-id": int(partition_id),
                }
                content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
                return Message(content=content, reply_to=msg)

            else:
                raise ValueError(f"Unknown FedXGBllr phase='{phase}'")

        # ============================================================
        # Existing XGBoost branch: Bagging / Cyclic
        # ============================================================
        if strategy_name == "fedxgbbagging":
            local_model_bytes = xgb_train_one_client(
                global_model_bytes=global_model_bytes,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
                num_classes=num_classes,
                num_local_round=local_epochs,
            )
        elif strategy_name == "fedxgbcyclic":
            local_model_bytes = xgb_train_one_client_cyclic(
                global_model_bytes=global_model_bytes,
                X_train=X_train,
                y_train=y_train,
                cfg=cfg,
                num_classes=num_classes,
                num_local_round=local_epochs,
            )
        else:
            raise ValueError(
                f"Unsupported strategy-name='{strategy_name}' for model-name='xgboost'. "
                f"Use 'fedxgbbagging', 'fedxgbcyclic', or 'fedxgbllr'."
            )

        t1 = time.perf_counter()
        monitor.stop()
        resource_stats = monitor.summary()

        cpu1, ram1 = get_cpu_ram_percent()
        temp1 = _try_read_pi_temp_c()
        tx1, rx1 = get_net_bytes()

        log_round({
            "exp_id": str(_k(cfg, "exp-id", "")),
            "client_id": partition_id,
            "model": model_name,
            "mode": partition_mode,
            "alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": None,
            "local_samples": int(len(y_train)),
            "train_time_s": float(t1 - t0),
            **resource_stats,
            "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
            "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and tx1 >= 0) else None,
            "param_count": None,
            "model_size_kib": float(len(local_model_bytes) / 1024.0),
        })

        model_np = np.frombuffer(local_model_bytes, dtype=np.uint8)
        model_record = ArrayRecord([model_np])

        metrics = {
            "train_loss": 0.0,
            "num-examples": int(len(y_train)),
            "partition-id": int(partition_id),
        }
        content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    # ============================================================
    # Existing PyTorch branch
    # ============================================================
    lr = float(msg.content["config"]["lr"])
    proximal_mu = float(msg.content["config"].get("proximal_mu", 0.0))

    trainloader, _, class_weights = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
        num_classes=num_classes,
    )

    model = build_model(model_name, num_features, num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    global_params = {k: v.detach().clone() for k, v in model.state_dict().items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss = train_fn(
        model,
        trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
        num_classes=num_classes,
        class_weights=class_weights,
        proximal_mu=proximal_mu,
        global_params=global_params,
        loss_type=str(_k(cfg, "loss-type", "ce")),
        loss_label_smoothing=float(_k(cfg, "loss-label-smoothing", 0.0)),
        focal_gamma=float(_k(cfg, "focal-gamma", 2.0)),
    )

    t1 = time.perf_counter()

    monitor.stop()
    resource_stats = monitor.summary()

    cpu1, ram1 = get_cpu_ram_percent()
    temp1 = _try_read_pi_temp_c()
    tx1, rx1 = get_net_bytes()

    param_count = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    log_round({
        "exp_id": str(_k(cfg, "exp-id", "")),
        "client_id": partition_id,
        "model": model_name,
        "mode": partition_mode,
        "alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": float(lr),
        "local_samples": int(len(trainloader.dataset)),
        "train_time_s": float(t1 - t0),
        **resource_stats,
        "net_tx_bytes_delta": (tx1 - tx0) if (tx0 >= 0 and tx1 >= 0) else None,
        "net_rx_bytes_delta": (rx1 - rx0) if (rx0 >= 0 and rx1 >= 0) else None,
        "param_count": int(param_count),
        "model_size_kib": float(model_bytes / 1024.0),
    })

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        "partition-id": int(partition_id),
    }
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    cfg = context.run_config

    model_name = str(_k(cfg, "model-name", "mlp")).lower().strip()
    partition_mode = str(_k(cfg, "partition-mode", "iid"))
    dirichlet_alpha = float(_k(cfg, "dirichlet-alpha", 0.5))
    batch_size = int(_k(cfg, "batch-size", 256))
    data_root = str(_k(cfg, "data-root", ""))

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    num_features, num_classes = get_num_features_classes_from_local_csv(
        partition_id=partition_id,
        num_partitions=num_partitions,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
    )

    # ============================================================
    # [ADD] LightGBM branch
    # ============================================================
    if model_name == "lightgbm":
        strategy_name = str(_k(cfg, "strategy-name", "fedxgbbagging")).lower().strip()

        _, _, X_val, y_val = load_data_numpy(
            partition_id=partition_id,
            num_partitions=num_partitions,
            mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            data_root=data_root,
        )

        global_model_bytes = b""
        if "arrays" in msg.content and "0" in msg.content["arrays"]:
            global_model_bytes = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # ------------------------------------------------------------
        # [ADD] LightGBM evaluation branch
        # ------------------------------------------------------------
        if strategy_name == "fedxgbllr":
            phase = str(msg.content["config"].get("phase", "eval_meta_xgbllr"))
            if phase != "eval_meta_xgbllr":
                raise ValueError(f"Unknown FedXGBllr eval phase='{phase}'")

            xgb_models_json = str(msg.content["config"]["xgb_models_json"])
            xgb_model_list = xgb_models_from_json(xgb_models_json)

            meta_features = build_xgbllr_features_from_models(
                xgb_model_list, X_val, cfg, num_classes
            )

            meta_model = build_xgbllr_meta_model(
                num_models=len(xgb_model_list),
                num_classes=num_classes,
            )
            meta_model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            meta_model.to(device)

            eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_xgbllr_meta(
                model=meta_model,
                features_np=meta_features,
                labels_np=y_val,
                batch_size=batch_size,
                device=device,
                num_classes=num_classes,
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
                "num-examples": int(len(y_val)),
            }

            content = RecordDict({"metrics": MetricRecord(metrics)})
            return Message(content=content, reply_to=msg)

        # ------------------------------------------------------------
        # [ADD] FedLGB evaluation branch
        # ------------------------------------------------------------
        local_model_bytes = global_model_bytes
        model = lgb.Booster(model_file=bytearray(local_model_bytes))

        eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = lgb_evaluate_bytes(
            model=model,
            X=X_val,
            y=y_val,
            cfg=cfg,
            num_classes=num_classes,
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
            "num-examples": int(len(y_val)),
        }

        content = RecordDict({"metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)
    
# ============================================================
# [ADD] XGBoost branch
# ============================================================

    if model_name == "xgboost":
        strategy_name = str(_k(cfg, "strategy-name", "fedxgbbagging")).lower().strip()

        _, _, X_val, y_val = load_data_numpy(
            partition_id=partition_id,
            num_partitions=num_partitions,
            mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            data_root=data_root,
        )

        # ============================================================
        # [ADD] FedXGBllr evaluation
        # ============================================================
        if strategy_name == "fedxgbllr":
            phase = str(msg.content["config"].get("phase", "eval_meta_xgbllr"))
            if phase != "eval_meta_xgbllr":
                raise ValueError(f"Unknown FedXGBllr eval phase='{phase}'")

            xgb_models_json = str(msg.content["config"]["xgb_models_json"])
            xgb_model_list = xgb_models_from_json(xgb_models_json)

            meta_features = build_xgbllr_features_from_models(
                xgb_model_list, X_val, cfg, num_classes
            )

            meta_model = build_xgbllr_meta_model(
                num_models=len(xgb_model_list),
                num_classes=num_classes,
            )
            meta_model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            meta_model.to(device)

            eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_xgbllr_meta(
                model=meta_model,
                features_np=meta_features,
                labels_np=y_val,
                batch_size=batch_size,
                device=device,
                num_classes=num_classes,
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
                "num-examples": int(len(y_val)),
            }

            content = RecordDict({"metrics": MetricRecord(metrics)})
            return Message(content=content, reply_to=msg)

        # ============================================================
        # Existing XGBoost evaluation: Bagging / Cyclic
        # ============================================================
        global_model_bytes = b""
        if "arrays" in msg.content and "0" in msg.content["arrays"]:
            global_model_bytes = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = xgb_evaluate_bytes(
            model_bytes=global_model_bytes,
            X=X_val,
            y=y_val,
            cfg=cfg,
            num_classes=num_classes,
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
            "num-examples": int(len(y_val)),
        }

        content = RecordDict({"metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    # ============================================================
    # Existing PyTorch branch
    # ============================================================
    model = build_model(model_name, num_features, num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, valloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        data_root=data_root,
    )

    eval_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
        model, valloader, device, num_classes=num_classes, criterion=None
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

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
