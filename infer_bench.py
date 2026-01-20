import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddos_attack.task import build_model, CSVDataset
from ddos_attack.bench_pi import get_cpu_ram_percent, _try_read_pi_temp_c, log_round


def load_bench(csv_path: str, batch_size: int):
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])

    y_raw = df[label_col].astype(str).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy().astype("float32")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    # local mapping only for shape; weights already compatible via num_features/num_classes
    uniq = sorted(set(y_raw.tolist()))
    m = {lab: i for i, lab in enumerate(uniq)}
    y = np.array([m[lab] for lab in y_raw], dtype="int64")

    ds = CSVDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False), int(X.shape[1]), int(y.max()) + 1


def benchmark_ms_per_sample(model, loader, device, warmup_batches=2, timed_batches=10):
    model.eval()
    model.to(device)

    # warmup
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["features"].to(device)
            _ = model(x)
            if i + 1 >= warmup_batches:
                break

    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["features"].to(device)
            _ = model(x)
            total += int(x.shape[0])
            if i + 1 >= timed_batches:
                break
    t1 = time.perf_counter()

    if total <= 0:
        return float("nan")
    return float((t1 - t0) * 1000.0 / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, choices=["mlp", "cnn1d", "cnn_bilstm"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--timed-batches", type=int, default=10)
    ap.add_argument("--client-id", type=int, default=-1)  # để gộp 7 Pi
    ap.add_argument("--exp-id", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    loader, num_features, num_classes = load_bench(args.data, args.batch_size)

    model = build_model(args.model_name, num_features, num_classes)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd)

    ms_list = []
    for r in range(args.repeat):
        cpu0, ram0 = get_cpu_ram_percent()
        temp0 = _try_read_pi_temp_c()

        ms = benchmark_ms_per_sample(
            model, loader, device,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
        )
        ms_list.append(ms)

        cpu1, ram1 = get_cpu_ram_percent()
        temp1 = _try_read_pi_temp_c()

        # append one record
        log_round({
            "type": "inference",
            "exp_id": args.exp_id,
            "client_id": args.client_id,
            "model": args.model_name,
            "batch_size": args.batch_size,
            "repeat": r,
            "ms_per_sample": float(ms),
            "cpu_percent_before": cpu0,
            "ram_percent_before": ram0,
            "temp_c_before": temp0,
            "cpu_percent_after": cpu1,
            "ram_percent_after": ram1,
            "temp_c_after": temp1,
            "data_file": str(Path(args.data).name),
        }, filename="pi_infer_benchmark.jsonl")

    ms_arr = np.array([x for x in ms_list if not np.isnan(x)], dtype=np.float32)
    if len(ms_arr) > 0:
        print(f"[{args.model_name}] ms/sample median={float(np.median(ms_arr)):.3f} mean={float(np.mean(ms_arr)):.3f} std={float(np.std(ms_arr)):.3f}")
    print("Saved to ~/projects/ddos-attack/fl_client_outputs/pi_infer_benchmark.jsonl")


if __name__ == "__main__":
    main()


# source ~/projects/fl_env/bin/activate
# cd ~/projects/ddos-attack
# python -m ddos_attack.infer_bench \
#   --model-name mlp \
#   --weights ~/projects/ddos-attack/ddos_models/final_model_mlp.pt \
#   --data ~/projects/ddos-attack/data/infer_bench.csv \
#   --repeat 3 \
#   --batch-size 256 \
#   --client-id 0 \
#   --exp-id exp_mlp_iid_N7