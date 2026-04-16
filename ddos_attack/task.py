"""ddos-attack: Flower / PyTorch app for DDoS detection (CSV).
- REAL FL: each client reads its own local CSV partition from data-root.
- IMPORTANT: CSVs are already MinMax-scaled -> DO NOT scale again.
- IMPORTANT: label mapping is REQUIRED and must be consistent across all nodes.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import base64

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Tuple, Dict, Any


# ============================================================
# 0) Strict label map (REQUIRED)
# ============================================================

_label_to_int: dict[str, int] | None = None
_int_to_label: list[str] | None = None

# ============================================================
# [ADD] Lazy import for XGBoost
# ============================================================

def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "XGBoost is required but not installed in the Flower runtime environment.\n"
            "Fix:\n"
            "  1) Add xgboost to [project].dependencies in pyproject.toml\n"
            "  2) Bump project.version\n"
            "  3) Re-run `flwr run ...` so Flower rebuilds the app bundle\n"
            "  4) Or install manually on each Pi: pip install xgboost\n"
        ) from e


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


# ----------------------------
# Utility blocks
# ----------------------------
def _get_act(act: str):
    act = (act or "relu").lower().strip()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act in ["silu", "swish"]:
        return nn.SiLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {act}")


class MLPBlock(nn.Module):
    """Linear -> Norm -> Act -> Dropout"""
    def __init__(self, in_dim, out_dim, act="relu", dropout=0.1, norm="layer"):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

        norm = (norm or "layer").lower().strip()
        if norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()

        self.act = _get_act(act)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# ============================================================
# Logistic Regression (FL-compatible)
# ============================================================
class LogisticRegressionNet(nn.Module):
    """
    Logistic Regression implementation in PyTorch.
    This is effectively a single linear layer, making it very 
    lightweight for Pi4 and compatible with FedAvg/FedProx.
    """
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # CrossEntropyLoss trong PyTorch sẽ tự tính Softmax, 
        # nên ở đây chỉ cần trả về kết quả Linear.
        return self.linear(x)

# ============================================================
# 1) MLPNet (FL-MLPNet-Lite)
#    - LN-based MLP, FL-friendly
# ============================================================

class MLPNet(nn.Module):
    """
    Paper model: FL-MLPNet-Lite
    Default: (128, 64, 32) + LayerNorm + Dropout
    """
    def __init__(self, num_features: int, num_classes: int,
                 dropout: float = 0.1, act: str = "relu", norm: str = "layer"):
        super().__init__()
        self.backbone = nn.Sequential(
            MLPBlock(num_features, 128, act=act, dropout=dropout, norm=norm),
            MLPBlock(128, 64, act=act, dropout=dropout, norm=norm),
            MLPBlock(64, 32, act=act, dropout=dropout, norm=norm),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)
    
# ============================================================
# 3) TabResNet (FL-TabResNet-Lite v2, ResMLP-style)
#    - Stronger residual blocks: 2x (LN->Linear->Act->Drop)
#    - More stable under non-IID
# ============================================================
class ResBlock(nn.Module):
    """
    ResMLP-style block:
      h = Linear(LN(x)) -> Act -> Drop
      h = Linear(LN(h)) -> Act -> Drop
      out = x + h
    """
    def __init__(self, dim=128, dropout=0.10, act: str = "relu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = _get_act(act)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.fc1(self.ln1(x))
        h = self.drop(self.act(h))
        h = self.fc2(self.ln2(h))
        h = self.drop(self.act(h))
        return x + h


class TabResNet(nn.Module):
    """
    Paper model: FL-TabResNet-Lite (v2)
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        dim: int = 128,
        depth: int = 3,
        dropout: float = 0.10,
        act: str = "relu",
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.LayerNorm(dim),
            _get_act(act),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )
        self.blocks = nn.Sequential(*[ResBlock(dim=dim, dropout=dropout, act=act) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_model(model_name: str, num_features: int, num_classes: int) -> nn.Module:
    mn = model_name.lower().strip()
    if mn == "logistic":
        return LogisticRegressionNet(num_features, num_classes)
    if mn == "mlp":
        return MLPNet(num_features, num_classes)
    if mn in ["tab-res-net", "tab_res_net", "tabresnet"]:
        return TabResNet(num_features, num_classes)
    raise ValueError(f"Unknown model_name='{model_name}'. Use one of: logistic, mlp, tab-res-net")


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
    num_classes: int = 13,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    local_csv = _local_client_csv_path(int(partition_id), int(num_partitions), mode, float(dirichlet_alpha), data_root)
    if not local_csv.exists():
        raise FileNotFoundError(f"Client CSV not found: {local_csv}")

    Xc, yc = _load_csv(str(local_csv))
    
    # Tính toán trọng số lớp một lần duy nhất tại đây
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weights_from_labels(yc, num_classes).to(device)

    n = len(Xc)
    rng = np.random.RandomState(100 + int(partition_id))
    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))

    X_train, y_train = Xc[idx[:split]], yc[idx[:split]]
    X_val, y_val = Xc[idx[split:]], yc[idx[split:]]

    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    train_loader = DataLoader(CSVDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(CSVDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, class_weights

# ============================================================
# [ADD] XGBoost utilities
# ============================================================

def load_data_numpy(
    partition_id: int,
    num_partitions: int,
    mode: str = "iid",
    dirichlet_alpha: float = 0.5,
    data_root: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local train/val split as numpy arrays for XGBoost."""
    local_csv = _local_client_csv_path(
        int(partition_id), int(num_partitions), mode, float(dirichlet_alpha), data_root
    )
    if not local_csv.exists():
        raise FileNotFoundError(f"Client CSV not found: {local_csv}")

    Xc, yc = _load_csv(str(local_csv))

    n = len(Xc)
    rng = np.random.RandomState(100 + int(partition_id))
    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))

    X_train, y_train = Xc[idx[:split]], yc[idx[:split]]
    X_val, y_val = Xc[idx[split:]], yc[idx[split:]]

    if len(X_val) == 0:
        X_val, y_val = X_train, y_train

    return X_train, y_train, X_val, y_val


def load_centralized_test_numpy(test_csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(test_csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"server-test-csv not found: {p}")
    return _load_csv(str(p))


def build_xgb_params_from_cfg(cfg: dict, num_classes: int) -> Dict[str, Any]:
    objective = str(cfg.get("xgb-objective", "multi:softprob"))
    params: Dict[str, Any] = {
        "eta": float(cfg.get("xgb-eta", 0.15)),
        "max_depth": int(cfg.get("xgb-max-depth", 4)),
        "subsample": float(cfg.get("xgb-subsample", 0.8)),
        "colsample_bytree": float(cfg.get("xgb-colsample-bytree", 0.8)),
        "max_bin": int(cfg.get("xgb-max-bin", 63)),
        "nthread": int(cfg.get("xgb-nthread", 2)),
        "tree_method": str(cfg.get("xgb-tree-method", "hist")),
        "objective": objective,
        "eval_metric": str(cfg.get("xgb-eval-metric", "mlogloss")),
        "verbosity": 0,
    }
    if objective.startswith("multi:"):
        params["num_class"] = int(num_classes)
    return params


def xgb_train_one_client(
    global_model_bytes: bytes | bytearray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    num_classes: int,
    num_local_round: int,
) -> bytes:
    """Train local XGBoost and return ONLY the local trees to send to server."""
    xgb = _require_xgboost()
    params = build_xgb_params_from_cfg(cfg, num_classes)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    if global_model_bytes is None or len(global_model_bytes) == 0:
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(num_local_round),
        )
    else:
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(global_model_bytes))

        for _ in range(int(num_local_round)):
            bst.update(dtrain, bst.num_boosted_rounds())

        start = max(0, bst.num_boosted_rounds() - int(num_local_round))
        bst = bst[start:bst.num_boosted_rounds()]

    local_model = bst.save_raw("json")
    return local_model


def xgb_predict_classes(
    model_bytes: bytes | bytearray,
    X: np.ndarray,
    cfg: dict,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Safe prediction for XGBoost Booster bytes.
    If model is empty / invalid, return fallback predictions instead of crashing.
    """
    xgb = _require_xgboost()

    # [ADD] empty model guard
    if model_bytes is None or len(model_bytes) == 0:
        pred = np.zeros(len(X), dtype=np.int64)
        probs = np.zeros((len(X), num_classes), dtype=np.float32)
        probs[:, 0] = 1.0
        return pred, probs

    params = build_xgb_params_from_cfg(cfg, num_classes)

    try:
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(model_bytes))

        # [ADD] extra safety: some boosters can load but still be unusable
        dmat = xgb.DMatrix(X)
        probs = bst.predict(dmat)

        if probs.ndim == 1:
            pred = (probs >= 0.5).astype(np.int64)
            probs_2d = np.stack([1.0 - probs, probs], axis=1)
            return pred, probs_2d

        pred = np.argmax(probs, axis=1).astype(np.int64)
        return pred, probs

    except Exception as e:
        # [ADD] fallback instead of crashing client
        print(f"[WARN] xgb_predict_classes fallback due to invalid booster: {e}")
        pred = np.zeros(len(X), dtype=np.int64)
        probs = np.zeros((len(X), num_classes), dtype=np.float32)
        probs[:, 0] = 1.0
        return pred, probs

# ============================================================
# [ADD] XGBoost train for FedXgbCyclic
# ============================================================

def xgb_train_one_client_cyclic(
    global_model_bytes: bytes | bytearray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    num_classes: int,
    num_local_round: int,
) -> bytes:
    """
    Train local XGBoost for FedXgbCyclic and return the FULL updated model.
    Difference from bagging:
      - Bagging returns only newly added local trees
      - Cyclic returns the full updated booster
    """
    xgb = _require_xgboost()
    params = build_xgb_params_from_cfg(cfg, num_classes)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Round 1 / empty global model
    if global_model_bytes is None or len(global_model_bytes) == 0:
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(num_local_round),
        )
        return bst.save_raw("json")

    # Continue training from received global booster
    bst = xgb.Booster(params=params)
    bst.load_model(bytearray(global_model_bytes))

    for _ in range(int(num_local_round)):
        bst.update(dtrain, bst.num_boosted_rounds())

    # [IMPORTANT] return FULL updated model for cyclic
    return bst.save_raw("json")

def xgb_evaluate_bytes(
    model_bytes: bytes | bytearray,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    num_classes: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Safe evaluation for XGBoost model bytes.
    Never crash on empty/invalid model.
    """
    pred, probs = xgb_predict_classes(model_bytes, X, cfg, num_classes)

    eps = 1e-12

    try:
        if probs.ndim == 2:
            row_idx = np.arange(len(y))
            clipped = np.clip(probs[row_idx, y], eps, 1.0)
            loss = float(-np.mean(np.log(clipped)))
        else:
            p = np.clip(probs, eps, 1.0 - eps)
            loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    except Exception:
        # [ADD] fallback loss
        loss = 999.0

    y_true_t = torch.from_numpy(y.astype(np.int64))
    y_pred_t = torch.from_numpy(pred.astype(np.int64))

    acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w = multiclass_metrics(
        y_true_t, y_pred_t, num_classes
    )
    return loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w

# ============================================================
# [ADD] FedXGBllr-style practical meta learner
# ============================================================

class FedXGBllrMetaCNN(nn.Module):
    """
    Flower-compatible practical FedXGBllr meta learner.

    Input:
        x: [B, K, C]
        - B: batch size
        - K: number of client XGBoost ensembles
        - C: number of classes

    We learn a lightweight 1D-CNN over stacked client-ensemble predictions.
    """
    def __init__(self, num_models: int, num_classes: int, hidden_channels: int = 32):
        super().__init__()
        self.num_models = int(num_models)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)

        self.conv = nn.Conv1d(
            in_channels=self.num_classes,
            out_channels=self.hidden_channels,
            kernel_size=self.num_models,
            stride=self.num_models,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, C] -> [B, C, K]
        if x.dim() != 3:
            raise ValueError(f"FedXGBllrMetaCNN expects [B, K, C], got shape={tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        x = self.conv(x)         # [B, hidden_channels, 1]
        x = self.act(x)
        x = x.squeeze(-1)        # [B, hidden_channels]
        x = self.fc(x)           # [B, C]
        return x


def build_xgbllr_meta_model(num_models: int, num_classes: int) -> nn.Module:
    return FedXGBllrMetaCNN(num_models=num_models, num_classes=num_classes, hidden_channels=32)


def xgb_models_to_json(model_bytes_list: list[bytes | bytearray]) -> str:
    payload = []
    for b in model_bytes_list:
        payload.append(base64.b64encode(bytes(b)).decode("utf-8"))
    return json.dumps(payload)


def xgb_models_from_json(payload: str) -> list[bytes]:
    if payload is None or str(payload).strip() == "":
        return []
    arr = json.loads(payload)
    return [base64.b64decode(x.encode("utf-8")) for x in arr]


def build_xgbllr_features_from_models(
    model_bytes_list: list[bytes | bytearray],
    X: np.ndarray,
    cfg: dict,
    num_classes: int,
) -> np.ndarray:
    """
    Build fixed meta-features from K frozen client XGBoost ensembles.

    Output shape:
        [N, K, C]
    """
    if len(model_bytes_list) == 0:
        raise ValueError("No XGBoost ensemble bytes provided to build FedXGBllr features.")

    feats = []
    for model_bytes in model_bytes_list:
        _, probs = xgb_predict_classes(model_bytes, X, cfg, num_classes)

        # Ensure [N, C]
        if probs.ndim == 1:
            probs = np.stack([1.0 - probs, probs], axis=1)

        if probs.shape[1] != num_classes:
            fixed = np.zeros((len(X), num_classes), dtype=np.float32)
            width = min(num_classes, probs.shape[1])
            fixed[:, :width] = probs[:, :width]
            probs = fixed

        feats.append(probs.astype(np.float32))

    return np.stack(feats, axis=1)  # [N, K, C]


def train_xgbllr_meta(
    model: nn.Module,
    features_np: np.ndarray,
    labels_np: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.to(device)

    x = torch.from_numpy(features_np.astype("float32"))
    y = torch.from_numpy(labels_np.astype("int64"))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    cw = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    model.train()
    total_loss = 0.0
    batch_count = 0

    for _ in range(int(epochs)):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

    return total_loss / max(batch_count, 1)


def test_xgbllr_meta(
    model: nn.Module,
    features_np: np.ndarray,
    labels_np: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    model.to(device)

    x = torch.from_numpy(features_np.astype("float32"))
    y = torch.from_numpy(labels_np.astype("int64"))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += float(loss.item())
            batch_count += 1

            pred = torch.argmax(logits, dim=1)
            all_true.append(yb.detach().cpu())
            all_pred.append(pred.detach().cpu())

    avg_loss = total_loss / max(batch_count, 1)
    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    (acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w) = multiclass_metrics(
        y_true, y_pred, num_classes
    )
    return avg_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w

# ============================================================
# 5) Training/Eval + metrics
# ============================================================

def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

# ============================
# [ADD] Loss factory (CE / Focal)
# ============================

class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss.
    alpha: Tensor[C] or None
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0.0:
            n_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_probs).sum(dim=1)
            pt = (probs * true_dist).sum(dim=1)
        else:
            ce = F.nll_loss(log_probs, target, reduction="none")
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            at = self.alpha.gather(0, target)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_criterion(
    loss_type: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    lt = (loss_type or "ce").lower().strip()

    if lt in ["ce", "cross_entropy", "cross-entropy"]:
        # CrossEntropyLoss supports label_smoothing in recent PyTorch
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))

    if lt in ["focal", "focal_loss", "focal-loss"]:
        return FocalLoss(alpha=class_weights, gamma=float(focal_gamma), label_smoothing=float(label_smoothing))

    raise ValueError(f"Unknown loss_type='{loss_type}'. Use: ce | focal")


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    num_classes: int,
    class_weights: torch.Tensor,
    proximal_mu: float = 0.0,
    global_params: dict | None = None,
    loss_type: str = "ce",
    loss_label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    ):
    net.to(device)

    # Sử dụng trọng số đã truyền vào để tạo criterion
    criterion = make_criterion(
        loss_type=loss_type,
        class_weights=class_weights, 
        label_smoothing=loss_label_smoothing,
        focal_gamma=focal_gamma,
    )

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
            
            if proximal_mu > 0.0 and global_params is not None:
                prox_term = 0.0
                for name, p in net.named_parameters():
                    p0 = global_params[name]
                    prox_term += torch.sum((p - p0) ** 2)
                loss = loss + 0.5 * proximal_mu * prox_term

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
