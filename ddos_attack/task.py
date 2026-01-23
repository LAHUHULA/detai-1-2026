"""ddos-attack: Flower / PyTorch app for DDoS detection (CSV).
- REAL FL: each client reads its own local CSV partition from data-root.
- IMPORTANT: CSVs are already MinMax-scaled -> DO NOT scale again.
- IMPORTANT: label mapping is REQUIRED and must be consistent across all nodes.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0) Strict label map (REQUIRED)
# ============================================================

_label_to_int: dict[str, int] | None = None
_int_to_label: list[str] | None = None


def _load_label_map_required() -> None:
    """Load label mapping from env var DDOS_LABEL_MAP (REQUIRED)."""
    global _label_to_int, _int_to_label

    if _label_to_int is not None:
        return

    lm_path = os.environ.get("DDOS_LABEL_MAP", "").strip()
    if not lm_path:
        raise RuntimeError(
            "DDOS_LABEL_MAP is REQUIRED but not set.\n"
            "Set it to an absolute path of label_map.json.\n"
            "Example (Linux):  export DDOS_LABEL_MAP=/home/ras-pi/ddos_data/label_map.json\n"
            "Example (Windows PowerShell): $env:DDOS_LABEL_MAP='C:\\ddos_data\\label_map.json'"
        )

    p = Path(lm_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"DDOS_LABEL_MAP not found: {p}")

    obj = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Invalid label_map.json format (must be non-empty dict): {p}")

    # validate integer values + build inverse list
    label_to_int: dict[str, int] = {}
    max_id = -1
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError(f"label_map key must be string, got {type(k)}")
        if not isinstance(v, int):
            raise ValueError(f"label_map value must be int, label={k}, got {type(v)}")
        if v < 0:
            raise ValueError(f"label_map value must be >=0, label={k}, got {v}")
        label_to_int[k] = v
        max_id = max(max_id, v)

    inv: list[str] = [""] * (max_id + 1)
    for lab, idx in label_to_int.items():
        inv[idx] = lab

    if any(x == "" for x in inv):
        # allow gaps? -> not recommended, fail fast
        missing = [i for i, x in enumerate(inv) if x == ""]
        raise ValueError(f"label_map has missing indices: {missing[:20]}... Fix label_map.json.")

    _label_to_int = label_to_int
    _int_to_label = inv


# ============================================================
# 1) Models
# ============================================================

class MLPNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.30)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, F]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [B, 64]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNNBiLSTMNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)            # [B, 1, F]
        x = F.relu(self.conv1(x))     # [B, 64, F]
        x = F.relu(self.conv2(x))     # [B, 128, F]
        x = self.pool(x)              # [B, 128, F/2]
        x = x.permute(0, 2, 1)        # [B, F/2, 128]
        out, _ = self.lstm(x)         # [B, F/2, 128]
        out = out[:, -1, :]           # [B, 128]
        out = self.dropout(out)
        return self.fc(out)


def build_model(model_name: str, num_features: int, num_classes: int) -> nn.Module:
    mn = model_name.lower().strip()
    if mn == "mlp":
        return MLPNet(num_features, num_classes)
    if mn == "cnn1d":
        return CNN1DNet(num_features, num_classes)
    if mn in ["cnn_bilstm", "cnn-bilstm", "cnn_bi_lstm"]:
        return CNNBiLSTMNet(num_features, num_classes)
    raise ValueError(f"Unknown model_name='{model_name}'. Use one of: mlp, cnn1d, cnn_bilstm")


# ============================================================
# 2) Dataset
# ============================================================

class CSVDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(features.astype("float32"))
        self.y = torch.from_numpy(labels.astype("int64"))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"features": self.x[idx], "label": self.y[idx]}


# ============================================================
# 3) CSV loader (NO scaling; label_map REQUIRED)
# ============================================================

def _load_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV and encode labels using REQUIRED global mapping."""
    _load_label_map_required()
    assert _label_to_int is not None

    df = pd.read_csv(path)

    label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])

    y_raw = df[label_col].astype(str).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    y = np.array([_label_to_int.get(lab, -1) for lab in y_raw], dtype="int64")
    if (y < 0).any():
        unknown = sorted(set(y_raw[y < 0].tolist()))
        raise ValueError(
            f"Found labels not in label_map.json: {unknown[:20]}...\n"
            f"CSV={path}"
        )

    return X, y


# ============================================================
# 4) Local-client CSV utilities (remote federation)
# ============================================================

def _resolve_data_root(data_root: str | None) -> Path:
    """data_root should point to ddos_data directory which contains clients/ and infer_bench.csv."""
    if data_root and str(data_root).strip():
        return Path(data_root).expanduser().resolve()
    # fallback only for local dev
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def _local_client_csv_path(
    partition_id: int,
    num_partitions: int,
    mode: str,
    dirichlet_alpha: float,
    data_root: str | None,
) -> Path:
    root = _resolve_data_root(data_root)
    clients_dir = root / "clients"

    if mode == "iid":
        return clients_dir / f"iid_n{num_partitions}" / f"client_{partition_id}.csv"

    if mode == "dirichlet":
        return clients_dir / f"dirichlet_a{dirichlet_alpha}_n{num_partitions}" / f"client_{partition_id}.csv"

    raise ValueError(f"Unknown partition mode: {mode}")


def get_num_features_classes_from_local_csv(
    partition_id: int,
    num_partitions: int,
    mode: str,
    dirichlet_alpha: float,
    data_root: str | None,
) -> Tuple[int, int]:
    p = _local_client_csv_path(partition_id, int(num_partitions), mode, dirichlet_alpha, data_root)
    X, y = _load_csv(str(p))
    return int(X.shape[1]), int(y.max()) + 1


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 256,
    mode: str = "iid",
    dirichlet_alpha: float = 0.5,
    data_root: str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Real FL: each client reads its own CSV partition:
      {data_root}/clients/iid/client_{id}.csv
      {data_root}/clients/dirichlet_a{alpha}/client_{id}.csv

    NOTE: CSV is already MinMax scaled -> do NOT apply scaling again.
    """
    local_csv = _local_client_csv_path(int(partition_id), int(num_partitions), mode, float(dirichlet_alpha), data_root)
    if not local_csv.exists():
        raise FileNotFoundError(f"Client CSV not found: {local_csv}")

    Xc, yc = _load_csv(str(local_csv))
    n = len(Xc)
    if n == 0:
        raise ValueError(f"Client {partition_id} has 0 samples: {local_csv}")

    rng = np.random.RandomState(100 + int(partition_id))
    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))

    X_train, y_train = Xc[idx[:split]], yc[idx[:split]]
    X_val, y_val = Xc[idx[split:]], yc[idx[split:]]

    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    train_loader = DataLoader(CSVDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(CSVDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ============================================================
# 5) Training/Eval + metrics
# ============================================================

def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train(net, trainloader, epochs, lr, device, num_classes: int):
    net.to(device)

    # local class weights from local train data (robust for non-IID)
    all_y = []
    for batch in trainloader:
        all_y.append(batch["label"].detach().cpu())
    y_np = torch.cat(all_y).numpy()

    w = compute_class_weights_from_labels(y_np, int(num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))

    net.train()
    total_loss = 0.0
    batch_count = 0

    for _ in range(int(epochs)):
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

    return total_loss / max(batch_count, 1)


def multiclass_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    tp = torch.diag(cm).to(torch.float32)
    fp = cm.sum(dim=0).to(torch.float32) - tp
    fn = cm.sum(dim=1).to(torch.float32) - tp
    support = cm.sum(dim=1).to(torch.float32)

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)

    precision = torch.nan_to_num(precision, nan=0.0)
    recall = torch.nan_to_num(recall, nan=0.0)
    f1 = torch.nan_to_num(f1, nan=0.0)

    accuracy = tp.sum() / torch.clamp(cm.sum().to(torch.float32), min=1.0)

    precision_macro = precision.mean().item()
    recall_macro = recall.mean().item()
    f1_macro = f1.mean().item()

    weights = support / torch.clamp(support.sum(), min=1.0)
    precision_weighted = (precision * weights).sum().item()
    recall_weighted = (recall * weights).sum().item()
    f1_weighted = (f1 * weights).sum().item()

    return (accuracy.item(),
            precision_macro, recall_macro, f1_macro,
            precision_weighted, recall_weighted, f1_weighted)


def test(net, testloader, device, num_classes: int, criterion=None):
    net.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    net.eval()
    total_loss = 0.0
    batch_count = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in testloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)
            logits = net(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item())
            batch_count += 1

            pred = torch.argmax(logits, dim=1)
            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    avg_loss = total_loss / max(batch_count, 1)
    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    (acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w) = multiclass_metrics(y_true, y_pred, num_classes)
    return avg_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w


def load_centralized_testloader(test_csv_path: str, batch_size: int = 256) -> DataLoader:
    """Server-side evaluation uses a global test CSV path (absolute path on PC)."""
    p = Path(test_csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"server-test-csv not found: {p}")
    X_test, y_test = _load_csv(str(p))
    return DataLoader(CSVDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
