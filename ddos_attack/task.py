"""ddos-attack: Flower / PyTorch app for DDoS detection (CSV)."""

from pathlib import Path
from typing import Tuple, Dict, Any
import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. Global config: paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # ddos-attack/
TRAIN_CSV_PATH = str(BASE_DIR / "data" / "train_final.csv")
TEST_CSV_PATH = str(BASE_DIR / "data" / "test_final.csv")

# IMPORTANT: Your CSV already MinMaxScaled => disable any internal scaling
APPLY_INTERNAL_SCALING = False  # keep False


# ============================================================
# 1. Models (3 options)
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
        self.dropout = nn.Dropout(0.3)
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
        x = x.unsqueeze(1)                  # [B, 1, F]
        x = F.relu(self.conv1(x))           # [B, 64, F]
        x = F.relu(self.conv2(x))           # [B, 128, F]
        x = self.pool(x)                    # [B, 128, F/2]

        x = x.permute(0, 2, 1)              # [B, F/2, 128]
        out, _ = self.lstm(x)               # [B, F/2, 128]
        out = out[:, -1, :]                 # [B, 128]

        out = self.dropout(out)
        return self.fc(out)


def build_model(model_name: str, num_features: int, num_classes: int) -> nn.Module:
    model_name = model_name.lower().strip()
    if model_name == "mlp":
        return MLPNet(num_features, num_classes)
    if model_name == "cnn1d":
        return CNN1DNet(num_features, num_classes)
    if model_name in ["cnn_bilstm", "cnn-bilstm", "cnn_bi_lstm"]:
        return CNNBiLSTMNet(num_features, num_classes)
    raise ValueError(f"Unknown model_name='{model_name}'. Use one of: mlp, cnn1d, cnn_bilstm")


# ============================================================
# 2. CSV Dataset Wrapper
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
# 3. Global caches
# ============================================================

_train_features: np.ndarray | None = None
_train_labels: np.ndarray | None = None

_test_features: np.ndarray | None = None
_test_labels: np.ndarray | None = None

_num_features: int | None = None
_num_classes: int | None = None

_label_to_int: dict | None = None
_int_to_label: list | None = None

_scaler_mean: np.ndarray | None = None
_scaler_std: np.ndarray | None = None

_class_weights_torch = None


# ============================================================
# 3.1 Label map (FIXED) helpers
# ============================================================

def _load_label_map_if_available():
    """Load a fixed label mapping from DDOS_LABEL_MAP to ensure consistency across devices."""
    global _label_to_int, _int_to_label

    if _label_to_int is not None:
        return

    p = os.environ.get("DDOS_LABEL_MAP", "").strip()
    if not p:
        return  # fallback to auto-fit on first loaded file (NOT recommended for real FL)

    mp = Path(p).expanduser().resolve()
    if not mp.exists():
        raise FileNotFoundError(f"DDOS_LABEL_MAP not found: {mp}")

    obj = json.loads(mp.read_text(encoding="utf-8"))
    _int_to_label = obj["labels"]
    _label_to_int = obj["map"]


# ============================================================
# 3.2 CSV loading (NO scaling here)
# ============================================================

def _load_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    global _label_to_int, _int_to_label

    _load_label_map_if_available()

    df = pd.read_csv(path)

    label_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[-1])
    y_raw = df[label_col].astype(str).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    # If mapping not provided, fallback auto-fit (simulation only)
    if _label_to_int is None:
        unique = sorted(set(y_raw.tolist()))
        _int_to_label = unique
        _label_to_int = {lab: i for i, lab in enumerate(_int_to_label)}

    y = np.array([_label_to_int.get(lab, -1) for lab in y_raw], dtype="int64")
    if (y < 0).any():
        unknown = sorted(set(y_raw[y < 0].tolist()))
        raise ValueError(f"Found labels not in fitted label mapping: {unknown[:10]} ...")

    return X, y


# ============================================================
# 3.3 Global load for simulation/inspection (NO scaling)
# ============================================================

def _ensure_data_loaded(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH):
    global _train_features, _train_labels, _test_features, _test_labels
    global _scaler_mean, _scaler_std, _num_features, _num_classes, _class_weights_torch

    if _train_features is None:
        X_train, y_train = _load_csv(train_path)

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # CSV already MinMaxScaled => do NOT scale again
        _scaler_mean = np.zeros((X_train.shape[1],), dtype=np.float32)
        _scaler_std = np.ones((X_train.shape[1],), dtype=np.float32)

        _train_features = X_train.astype("float32")
        _train_labels = y_train.astype("int64")

        _num_features = _train_features.shape[1]
        _num_classes = int(_train_labels.max()) + 1

        # global class weights (optional)
        counts = np.bincount(_train_labels, minlength=_num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        _class_weights_torch = torch.tensor(weights, dtype=torch.float32)

    if _test_features is None and Path(test_path).exists():
        X_test, y_test = _load_csv(test_path)
        _test_features = X_test.astype("float32")
        _test_labels = y_test.astype("int64")


def get_num_features_classes() -> Tuple[int, int]:
    _ensure_data_loaded()
    return int(_num_features), int(_num_classes)


def get_flower_dataset() -> Dict[str, Any]:
    _ensure_data_loaded()
    return {"features": _train_features, "label": _train_labels}


# ============================================================
# 4. REAL FL: local client CSV reading
# ============================================================

def _get_data_root_clients() -> Path:
    """Return the directory containing data/clients (outside .flwr/apps if DDOS_DATA_ROOT is set)."""
    data_root_env = os.environ.get("DDOS_DATA_ROOT", "").strip()
    if data_root_env:
        return Path(data_root_env).expanduser().resolve()
    # fallback: inside app package (may not have data)
    return (Path(__file__).resolve().parent.parent / "data" / "clients").resolve()


def get_num_features_classes_from_local_csv(partition_id: int, mode: str, dirichlet_alpha: float = 0.5):
    data_root = _get_data_root_clients()

    if mode == "iid":
        local_csv = data_root / "iid" / f"client_{partition_id}.csv"
    elif mode == "dirichlet":
        local_csv = data_root / f"dirichlet_a{str(dirichlet_alpha)}" / f"client_{partition_id}.csv"
    else:
        raise ValueError(f"Unknown mode={mode}")

    X, y = _load_csv(str(local_csv))
    return int(X.shape[1]), int(y.max()) + 1


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 256,
    mode: str = "iid",
    dirichlet_alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """
    REAL FL mode:
      - Read pre-split client CSV: already MinMaxScaled => DO NOT scale again.
    """
    data_root = _get_data_root_clients()

    if mode == "iid":
        local_csv = data_root / "iid" / f"client_{partition_id}.csv"
    elif mode == "dirichlet":
        local_csv = data_root / f"dirichlet_a{str(dirichlet_alpha)}" / f"client_{partition_id}.csv"
    else:
        raise ValueError(f"Unknown mode={mode}")

    if not local_csv.exists():
        raise FileNotFoundError(f"Client CSV not found: {local_csv}")

    Xc, yc = _load_csv(str(local_csv))
    Xc = Xc.astype("float32")  # already scaled (MinMaxScaler)

    n = len(Xc)
    if n == 0:
        raise ValueError(f"Client {partition_id} has 0 samples in {local_csv}")

    rng = np.random.RandomState(100 + partition_id)
    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))

    X_train, y_train = Xc[idx[:split]], yc[idx[:split]]
    X_val, y_val = Xc[idx[split:]], yc[idx[split:]]

    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    train_ds = CSVDataset(X_train, y_train)
    val_ds = CSVDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


# ============================================================
# 5. Loss weights + Train/Eval
# ============================================================

def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train(net, trainloader, epochs, lr, device, num_classes: int | None = None):
    net.to(device)

    global _class_weights_torch

    # If global weights exist (simulation), use them; else compute local weights
    if _class_weights_torch is not None:
        criterion = nn.CrossEntropyLoss(weight=_class_weights_torch.to(device))
    else:
        all_y = []
        for batch in trainloader:
            all_y.append(batch["label"].detach().cpu())
        y_np = torch.cat(all_y).numpy()
        if num_classes is None:
            num_classes = int(y_np.max()) + 1
        w = compute_class_weights_from_labels(y_np, int(num_classes)).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    total_loss = 0.0
    batch_count = 0

    for _ in range(epochs):
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

    return total_loss / max(batch_count, 1)


def multiclass_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

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

    return (
        accuracy.item(),
        precision_macro, recall_macro, f1_macro,
        precision_weighted, recall_weighted, f1_weighted
    )


def load_centralized_testloader(batch_size: int = 256) -> DataLoader:
    """
    Server-side evaluation:
      - Reads from env DDOS_SERVER_TEST_CSV if provided (absolute path).
      - Else tries package path data/test_final.csv.
    Note: No scaling applied (CSV already MinMaxScaled).
    """
    env_path = os.environ.get("DDOS_SERVER_TEST_CSV", "").strip()

    if env_path:
        test_path = Path(env_path).expanduser().resolve()
        if not test_path.exists():
            raise FileNotFoundError(f"DDOS_SERVER_TEST_CSV not found: {test_path}")
        X_test, y_test = _load_csv(str(test_path))
        ds = CSVDataset(X_test, y_test)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    test_path = Path(__file__).resolve().parent.parent / "data" / "test_final.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test CSV not found. Set env DDOS_SERVER_TEST_CSV to an absolute path. Tried: {test_path}"
        )

    X_test, y_test = _load_csv(str(test_path))
    ds = CSVDataset(X_test, y_test)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def test(net, testloader, device, num_classes: int, criterion=None):
    net.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    net.eval()
    total_loss = 0.0
    batch_count = 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in testloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            logits = net(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            batch_count += 1

            pred = torch.argmax(logits, dim=1)
            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    avg_loss = total_loss / max(batch_count, 1)

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    (acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w) = multiclass_metrics(y_true, y_pred, num_classes)

    return avg_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w
