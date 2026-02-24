import os
import json
import time
from pathlib import Path

import psutil


def _out_dir() -> Path:
    p = Path(os.environ.get("DDOS_CLIENT_OUT", "~/projects/ddos-attack/fl_client_outputs")).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cpu_ram_percent():
    cpu = psutil.cpu_percent(interval=0.3)
    ram = psutil.virtual_memory().percent
    return float(cpu), float(ram)


def _try_read_pi_temp_c():
    try:
        p = Path("/sys/class/thermal/thermal_zone0/temp")
        if p.exists():
            v = float(p.read_text().strip())
            return float(v / 1000.0)
    except Exception:
        pass
    return None


def get_net_bytes(iface: str | None = None):
    try:
        if iface:
            per = psutil.net_io_counters(pernic=True)
            if iface in per:
                c = per[iface]
                return int(c.bytes_sent), int(c.bytes_recv)
        c = psutil.net_io_counters()
        return int(c.bytes_sent), int(c.bytes_recv)
    except Exception:
        return -1, -1


# def log_round(record: dict, filename: str = "pi_train_benchmark.jsonl"):
#     out = _out_dir() / filename
#     record = dict(record)
#     record.setdefault("ts", time.time())
#     with out.open("a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_round(record: dict, filename: str | None = None):
    out_dir = _out_dir()
    record = dict(record)
    record.setdefault("ts", time.time())

    # Lấy exp_id để tạo file riêng
    exp_id = record.get("exp_id", "unknown_exp")

    # sanitize tên file (tránh ký tự lạ)
    safe_exp_id = str(exp_id).replace(" ", "_").replace("/", "_")

    # nếu không truyền filename → tự tạo theo exp_id
    if filename is None:
        filename = f"pi_train_benchmark_{safe_exp_id}.jsonl"

    out_path = out_dir / filename

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ================================
# Advanced Resource Monitor (for FL round)
# ================================

import threading
import statistics


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.cpu_vals = []
        self.ram_vals = []
        self.temp_vals = []

    def _collect(self):
        # Warmup call to avoid first 0.0 CPU reading
        psutil.cpu_percent(interval=None)

        while self.running:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

            temp = None
            try:
                p = Path("/sys/class/thermal/thermal_zone0/temp")
                if p.exists():
                    temp = float(p.read_text().strip()) / 1000.0
            except Exception:
                pass

            self.cpu_vals.append(float(cpu))
            self.ram_vals.append(float(ram))
            if temp is not None:
                self.temp_vals.append(float(temp))

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._collect)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()

    def _stats(self, arr):
        if not arr:
            return 0.0, 0.0, 0.0
        mean = float(statistics.mean(arr))
        std = float(statistics.stdev(arr)) if len(arr) > 1 else 0.0
        maxv = float(max(arr))
        return mean, std, maxv

    def summary(self):
        cpu_mean, cpu_std, cpu_max = self._stats(self.cpu_vals)
        ram_mean, ram_std, ram_max = self._stats(self.ram_vals)
        temp_mean, temp_std, temp_max = self._stats(self.temp_vals)

        return {
            "cpu_mean": cpu_mean,
            "cpu_std": cpu_std,
            "cpu_max": cpu_max,
            "ram_mean": ram_mean,
            "ram_std": ram_std,
            "ram_max": ram_max,
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "temp_max": temp_max,
        }
