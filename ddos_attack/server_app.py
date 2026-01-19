import time
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ddos_attack.task import build_model, load_centralized_testloader
from ddos_attack.task import test as test_fn

app = ServerApp()


def _get_run_cfg(context: Context):
    cfg = context.run_config
    return {
        "fraction_train": float(cfg.get("fraction_train", 1.0)),
        "num_rounds": int(cfg.get("num_server_rounds", 1)),
        "lr": float(cfg.get("lr", 1e-3)),
        "model_name": cfg.get("model_name", "mlp"),
        "num_clients": int(cfg.get("num_clients", 2)),
        "batch_size": int(cfg.get("batch_size", 256)),
        # IMPORTANT: must be consistent across clients
        "num_features": int(cfg.get("num_features", 20)),
        "num_classes": int(cfg.get("num_classes", 13)),
        "do_centralized_test": bool(cfg.get("do_centralized_test", True)),
    }


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_cfg = _get_run_cfg(context)

    # Build global model
    global_model = build_model(
        model_name=run_cfg["model_name"],
        num_features=run_cfg["num_features"],
        num_classes=run_cfg["num_classes"],
    )
    initial_arrays = ArrayRecord(global_model.state_dict())

    # FL Strategy
    strategy = FedAvg(fraction_train=run_cfg["fraction_train"])

    # Train
    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": run_cfg["lr"]}),
        num_rounds=run_cfg["num_rounds"],
    )
    t1 = time.perf_counter()

    # Save final model
    state_dict = result.arrays.to_torch_state_dict()
    out_path = f"final_model_{run_cfg['model_name']}.pt"
    torch.save(state_dict, out_path)
    print(f"\nSaved {out_path}")

    # Estimated logging
    total_time = t1 - t0
    avg_round_time = total_time / max(run_cfg["num_rounds"], 1)

    model_bytes = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            model_bytes += v.numel() * v.element_size()

    clients_per_round = max(1, int(round(run_cfg["fraction_train"] * run_cfg["num_clients"])))
    comm_overhead_per_round = model_bytes * 2 * clients_per_round  # download + upload

    print("\n=== FL LOGGING (estimated) ===")
    print(f"Total time (s): {total_time:.2f}")
    print(f"Avg round time (s): {avg_round_time:.2f}")
    print(f"Model size (MB): {model_bytes / (1024**2):.2f}")
    print(f"Estimated comm/round (MB): {comm_overhead_per_round / (1024**2):.2f}")

    # Optional: centralized test on server
    if not run_cfg["do_centralized_test"]:
        return

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = build_model(
            model_name=run_cfg["model_name"],
            num_features=run_cfg["num_features"],
            num_classes=run_cfg["num_classes"],
        ).to(device)

        model.load_state_dict(state_dict)

        testloader = load_centralized_testloader(batch_size=run_cfg["batch_size"])

        loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
            net=model,
            testloader=testloader,
            device=device,
            num_classes=run_cfg["num_classes"],
            criterion=None,  # default CrossEntropyLoss
        )

        print("\n=== CENTRALIZED TEST (test_final.csv) ===")
        print(f"loss={loss:.4f} | acc={acc:.4f}")
        print(f"Macro:    P={p_macro:.4f} | R={r_macro:.4f} | F1={f1_macro:.4f}")
        print(f"Weighted: P={p_w:.4f} | R={r_w:.4f} | F1={f1_w:.4f}")

    except Exception as e:
        print("\n[WARNING] Centralized test failed:", repr(e))
