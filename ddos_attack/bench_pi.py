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


def log_round(record: dict, filename: str = "pi_train_benchmark.jsonl"):
    out = _out_dir() / filename
    record = dict(record)
    record.setdefault("ts", time.time())
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
