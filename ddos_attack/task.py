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

class TinyMLP(nn.Module):
  def __init__(self, num_features:int, num_classes:int, dropout:float=0.15):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes)
    )

  def forward(self, x):
    return self.net(x)

# ============================================================
# 1) MLPNet (CHỈNH) - giảm dropout + thêm LayerNorm để ổn định FL
#    - Thay đổi chính:
#      [CHANGE] dropout: 0.3 -> 0.1 (hoặc 0.2)
#      [ADD] LayerNorm sau Linear
# ============================================================
class MLPNet(nn.Module):
    """
    MLP ổn định hơn trong Federated Learning (non-IID) nhờ LayerNorm.
    """
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.1):  # [CHANGE] default dropout
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),          # [ADD]
            nn.ReLU(),
            nn.Dropout(dropout),       # [CHANGE]

            nn.Linear(64, 64),
            nn.LayerNorm(64),          # [ADD]
            nn.ReLU(),
            nn.Dropout(dropout),       # [CHANGE]

            nn.Linear(64, 32),
            nn.LayerNorm(32),          # [ADD]
            nn.ReLU(),
            nn.Dropout(dropout),       # [CHANGE]

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================
# 2) DCN-Lite (CHỈNH) - giảm độ "gắt" của cross + dropout nhẹ sau cross
#    - Thay đổi chính:
#      [CHANGE] num_cross: 2 -> 1  (mặc định 1 cho ổn định)
#      [OPTION] dim: 128 -> 64 (nếu muốn ổn định hơn nữa)
#      [ADD] dropout sau mỗi cross layer output
# ============================================================
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Giữ init như bạn đang làm; nếu muốn ổn định hơn có thể đổi init Xavier,
        # nhưng ở đây mình không bắt buộc để bạn thay ít nhất.
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        wx = torch.sum(x * self.w, dim=1, keepdim=True)  # [B, 1]
        return x0 * wx + self.b + x


class DCNLiteNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dim: int = 128,
        num_cross: int = 1,          # [CHANGE] 2 -> 1 (ổn định hơn cho non-IID)
        dropout: float = 0.1,
        cross_dropout: float = 0.05  # [ADD] dropout nhẹ cho nhánh cross
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.LayerNorm(dim),
        )

        self.cross = nn.ModuleList([CrossLayer(dim) for _ in range(num_cross)])
        self.cross_drop = nn.Dropout(cross_dropout)  # [ADD]

        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(dim + 64, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.embed(x)  # [B, dim]
        x0 = x
        xc = x

        for layer in self.cross:
            xc = layer(x0, xc)
            xc = self.cross_drop(xc)  # [ADD] giúp nhánh cross bớt dao động

        xd = self.deep(x)
        out = torch.cat([xc, xd], dim=1)
        return self.fc(out)


# ============================================================
# 3) TabResNet (TUỲ CHỌN CHỈNH NHẸ) - tăng depth hoặc giảm dropout
#    - Thay đổi chính (tuỳ chọn):
#      [OPTION] depth: 3 -> 4 (tăng năng lực)
#      [OPTION] dropout: 0.1 -> 0.05 (nếu thấy học hơi ì)
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = self.ln2(self.fc2(h))
        return F.relu(x + h)


class TabResNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dim=128,
        depth=3,            # [OPTION] đổi thành 4 nếu muốn
        dropout=0.1         # [OPTION] đổi thành 0.05 nếu muốn mượt/ít regularize hơn
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(dim, dropout) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_model(model_name: str, num_features: int, num_classes: int) -> nn.Module:
    mn = model_name.lower().strip()
    if mn == "mlp":
        return MLPNet(num_features, num_classes)
    if mn == "dcn-lite":
        return DCNLiteNet(num_features, num_classes)
    if mn in ["tab-res-net", "tab_res_net", "tabresnet"]:
        return TabResNet(num_features, num_classes)
    if mn in ["TinyMLP", "tinymlp", "tiny-mlp"]:
        return TinyMLP(num_features, num_classes)
    raise ValueError(f"Unknown model_name='{model_name}'. Use one of: mlp, dcn-lite, tab-res-net, tinymlp")


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
    _load_label_map_required()
    assert _int_to_label is not None
    num_features = int(X.shape[1])
    num_classes = int(len(_int_to_label))
    return num_features, num_classes

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


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    num_classes: int,
    proximal_mu: float = 0.0,
    global_params: dict | None = None,
    ):
    net.to(device)

    # local class weights from local train data (robust for non-IID)
    all_y = []
    for batch in trainloader:
        all_y.append(batch["label"].detach().cpu())
    y_np = torch.cat(all_y).numpy()

    w = compute_class_weights_from_labels(y_np, int(num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    # criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(lr),
        weight_decay=1e-4
    )

    net.train()
    total_loss = 0.0
    batch_count = 0

    for _ in range(int(epochs)):
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(net(x), y)
            # =======================
            # FedProx proximal term
            # =======================
            if proximal_mu > 0.0 and global_params is not None:
                prox_term = 0.0
                for name, p in net.named_parameters():
                    p0 = global_params[name]
                    prox_term += torch.sum((p - p0) ** 2)
                loss = loss + 0.5 * proximal_mu * prox_term
            # =======================
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
