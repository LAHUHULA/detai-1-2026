import time
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ddos_attack.task import build_model, get_num_features_classes
from ddos_attack.task import load_centralized_testloader, test as test_fn
from ddos_attack.task import get_class_weight_criterion

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config["fraction_train"]
    num_rounds: int = context.run_config["num_server_rounds"]
    lr: float = context.run_config["lr"]
    

    model_name: str = context.run_config.get("model_name", "mlp")
    num_clients: int = int(context.run_config.get("num_clients", 10))  # add to pyproject
    batch_size: int = int(context.run_config.get("batch_size", 256))

    num_features: int = int(context.run_config.get("num_features", 40))
    num_classes: int = int(context.run_config.get("num_classes", 13))
    # num_features, num_classes = get_num_features_classes()

    global_model = build_model(model_name, num_features, num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_train=fraction_train)

    t0 = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    t1 = time.perf_counter()

    # Save final model
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"final_model_{model_name}.pt")
    print(f"\nSaved final_model_{model_name}.pt")

    # ---------------------------
    # Logging: time / model size / comm overhead (estimated)
    # ---------------------------
    total_time = t1 - t0
    avg_round_time = total_time / max(num_rounds, 1)

    # model size in bytes (float32 params)
    model_bytes = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            model_bytes += v.numel() * v.element_size()

    # estimated clients per round
    clients_per_round = max(1, int(round(fraction_train * num_clients)))
    comm_overhead_per_round = model_bytes * 2 * clients_per_round  # down + up

    print("\n=== FL LOGGING (estimated) ===")
    print(f"Total time (s): {total_time:.2f}")
    print(f"Avg round time (s): {avg_round_time:.2f}")
    print(f"Model size (MB): {model_bytes / (1024**2):.2f}")
    print(f"Estimated comm/round (MB): {comm_overhead_per_round / (1024**2):.2f}")

    # ---------------------------
    # Centralized evaluation on test_final.csv
    # ---------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_features, num_classes).to(device)
    model.load_state_dict(state_dict)

    testloader = load_centralized_testloader(batch_size=batch_size)
    criterion = get_class_weight_criterion(device=device)

    loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = test_fn(
        model, testloader, device, num_classes=num_classes, criterion=criterion
    )


    print("\n=== CENTRALIZED TEST (global test_final.csv) ===")
    print(f"loss={loss:.4f} | acc={acc:.4f}")
    print(f"Macro:    P={p_macro:.4f} | R={r_macro:.4f} | F1={f1_macro:.4f}")
    print(f"Weighted: P={p_w:.4f} | R={r_w:.4f} | F1={f1_w:.4f}")

