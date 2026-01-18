import os
import time
import json
from pathlib import Path

def _try_read_pi_temp_c() -> float | None:
    # Pi OS: vcgencmd
    try:
        import subprocess
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        # temp=48.2'C
        val = out.split("=")[1].replace("'C", "")
        return float(val)
    except Exception:
        return None

def get_cpu_ram_percent() -> tuple[float, float]:
    try:
        import psutil
        cpu = float(psutil.cpu_percent(interval=0.2))
        ram = float(psutil.virtual_memory().percent)
        return cpu, ram
    except Exception:
        return -1.0, -1.0

def get_net_bytes() -> tuple[int, int]:
    try:
        import psutil
        io = psutil.net_io_counters()
        return int(io.bytes_sent), int(io.bytes_recv)
    except Exception:
        return -1, -1

def ensure_log_dir() -> Path:
    p = Path("logs")
    p.mkdir(parents=True, exist_ok=True)
    return p

def log_round(payload: dict, filename: str = "pi_benchmark.jsonl") -> None:
    log_dir = ensure_log_dir()
    path = log_dir / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def infer_latency_ms_per_sample(model, sample_batch, device, repeats: int = 200) -> float:
    """
    Measure inference latency (ms/sample) on current device.
    sample_batch: tensor [B, F] (we will use batch_size=1 by default)
    """
    import torch

    model.eval()
    sample_batch = sample_batch.to(device)

    # warmup
    with torch.no_grad():
        for _ in range(30):
            _ = model(sample_batch)

    # timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(sample_batch)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    ms_per_call = total_ms / repeats
    # if batch_size>1, convert to per sample
    bsz = int(sample_batch.shape[0])
    return ms_per_call / max(bsz, 1)
