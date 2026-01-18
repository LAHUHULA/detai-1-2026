"""ddos-attack: Flower / PyTorch app for DDoS detection (CSV)."""

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Flower partitioners
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
)
from flwr_datasets.partitioner import IidPartitioner


# ============================================================
# 0. Global config: paths
# ============================================================
# TRAIN_CSV_PATH = "data/train_final.csv"
# TEST_CSV_PATH = "data/test_final.csv"
BASE_DIR = Path(__file__).resolve().parent.parent  # thư mục gốc project (ddos-attack/)
TRAIN_CSV_PATH = str(BASE_DIR / "data" / "train_final.csv")
TEST_CSV_PATH = str(BASE_DIR / "data" / "test_final.csv")


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

        # BiLSTM: input_size = 128 (channels after conv), seq_len ~ F/2
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
        # x: [B, F]
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

    raise ValueError(f"Unknown model-name='{model_name}'. Use one of: mlp, cnn1d, cnn_bilstm")


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
        return {
            "features": self.x[idx],
            "label": self.y[idx],
        }


# ============================================================
# 3. CSV Loading + Cache + Scaling
# ============================================================

_train_features: np.ndarray | None = None
_train_labels: np.ndarray | None = None

_test_features: np.ndarray | None = None
_test_labels: np.ndarray | None = None

_num_features: int | None = None
_num_classes: int | None = None

# Label encoder cache (string -> int)
_label_to_int: dict | None = None
_int_to_label: list | None = None

# Simple scaler cache (fit on train only, optional but recommended for NN)
_scaler_mean: np.ndarray | None = None
_scaler_std: np.ndarray | None = None

_num_features: int | None = None
_num_classes: int | None = None

_class_weights_torch = None



def _load_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    global _label_to_int, _int_to_label

    df = pd.read_csv(path)

    # Detect label column
    label_col = (
        "label"
        if "label" in df.columns
        else ("Label" if "Label" in df.columns else df.columns[-1])
    )

    y_raw = df[label_col].astype(str).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy()

    # Defensive cleanup
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    # If labels already look like integers, keep them
    # Otherwise, encode string labels -> int
    if _label_to_int is None:
        # Fit mapping on first call (train should be loaded first)
        unique = sorted(set(y_raw.tolist()))
        _int_to_label = unique
        _label_to_int = {lab: i for i, lab in enumerate(_int_to_label)}

    y = np.array([_label_to_int.get(lab, -1) for lab in y_raw], dtype="int64")
    if (y < 0).any():
        unknown = sorted(set(y_raw[y < 0].tolist()))
        raise ValueError(f"Found labels not in fitted label mapping: {unknown[:10]} ...")

    return X, y



def _fit_global_scaler(X: np.ndarray):
    """Simple StandardScaler implemented with numpy (fit on global train for simulation)."""
    global _scaler_mean, _scaler_std
    _scaler_mean = X.mean(axis=0)
    _scaler_std = X.std(axis=0)
    _scaler_std[_scaler_std == 0] = 1.0


def _transform_global_scaler(X: np.ndarray) -> np.ndarray:
    global _scaler_mean, _scaler_std
    return (X - _scaler_mean) / _scaler_std


def _ensure_data_loaded(train_path=TRAIN_CSV_PATH, test_path=TEST_CSV_PATH):
    global _train_features, _train_labels, _test_features, _test_labels
    global _scaler_mean, _scaler_std, _num_features, _num_classes

    if _train_features is None:
        X_train, y_train = _load_csv(train_path)

        # Shuffle once globally (reproducible)
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Fit scaler on TRAIN only (simple StandardScaler)
        _scaler_mean = X_train.mean(axis=0)
        _scaler_std = X_train.std(axis=0)
        _scaler_std[_scaler_std == 0] = 1.0

        X_train = (X_train - _scaler_mean) / _scaler_std

        _train_features = X_train
        _train_labels = y_train

        _num_features = _train_features.shape[1]
        _num_classes = int(_train_labels.max()) + 1

        # ---- compute class weights (GLOBAL) ----
        global _class_weights_torch
        counts = np.bincount(_train_labels, minlength=_num_classes).astype(np.float32)

        # avoid division by zero
        counts[counts == 0] = 1.0

        # inverse frequency weights
        weights = (counts.sum() / counts)

        # normalize (optional, helps stability)
        weights = weights / weights.mean()

        _class_weights_torch = torch.tensor(weights, dtype=torch.float32)


    if _test_features is None and Path(test_path).exists():
        X_test, y_test = _load_csv(test_path)
        X_test = (X_test - _scaler_mean) / _scaler_std

        _test_features = X_test
        _test_labels = y_test



def get_num_features_classes() -> Tuple[int, int]:
    _ensure_data_loaded()
    return int(_num_features), int(_num_classes)



# ============================================================
# 4. Flower-Compatible Dataset Format
# ============================================================

def get_flower_dataset() -> Dict[str, Any]:
    _ensure_data_loaded()
    return {
        "features": _train_features,
        "label": _train_labels,
    }


# ============================================================
# 5. Data Partitioning using Flower Partitioners
# ============================================================

_partitions_cache = {}  # key: (mode, alpha, num_partitions) -> list[np.ndarray indices]

def _build_partitions(mode: str, num_partitions: int, alpha: float = 0.5, seed: int = 42):
    _ensure_data_loaded()
    X = _train_features
    y = _train_labels
    rng = np.random.RandomState(seed)

    idx_all = rng.permutation(len(X))

    if mode == "iid":
        chunks = np.array_split(idx_all, num_partitions)
        return [c.astype(np.int64) for c in chunks]

    if mode == "dirichlet":
        # Robust Dirichlet partitioning (non-IID), returns INDEX arrays (int64)
        classes = np.unique(y)

        # indices per class
        idx_by_class = {c: np.where(y == c)[0] for c in classes}
        for c in classes:
            rng.shuffle(idx_by_class[c])

        # buckets of indices per client
        client_indices = [[] for _ in range(num_partitions)]

        for c in classes:
            idx_c = idx_by_class[c]
            n_c = len(idx_c)
            if n_c == 0:
                continue

            # Dirichlet proportions across clients
            proportions = rng.dirichlet(alpha * np.ones(num_partitions))

            # turn proportions into integer counts
            counts = (proportions * n_c).astype(int)

            # fix rounding so sum(counts) == n_c
            diff = n_c - counts.sum()
            while diff > 0:
                counts[rng.randint(0, num_partitions)] += 1
                diff -= 1
            while diff < 0:
                j = rng.randint(0, num_partitions)
                if counts[j] > 0:
                    counts[j] -= 1
                    diff += 1

            # assign slices to each client
            start = 0
            for client_id in range(num_partitions):
                end = start + counts[client_id]
                if end > start:
                    client_indices[client_id].extend(idx_c[start:end].tolist())
                start = end

        # shuffle per client
        for i in range(num_partitions):
            rng.shuffle(client_indices[i])

        # ensure no empty clients: steal from largest donors
        empty = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
        if empty:
            donors = sorted(range(num_partitions), key=lambda k: len(client_indices[k]), reverse=True)
            for ec in empty:
                donor = None
                for d in donors:
                    if len(client_indices[d]) > 1:  # don't make donor empty
                        donor = d
                        break
                if donor is None:
                    raise ValueError(
                        f"Cannot fix empty client {ec}. "
                        f"Dataset too small or num_partitions={num_partitions} too large."
                    )
                client_indices[ec].append(client_indices[donor].pop())

        # IMPORTANT: return index arrays, not (X_part, y_part)
        return [np.asarray(idxs, dtype=np.int64) for idxs in client_indices]



def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 256,
    mode: str = "iid",
    dirichlet_alpha: float = 0.5
) -> Tuple[DataLoader, DataLoader]:

    key = (mode, float(dirichlet_alpha), int(num_partitions))
    if key not in _partitions_cache:
        _partitions_cache[key] = _build_partitions(mode, num_partitions, alpha=dirichlet_alpha)

    idx_client = _partitions_cache[key][partition_id]

    Xc = _train_features[idx_client]
    yc = _train_labels[idx_client]

    n = len(Xc)
    if n == 0:
        raise ValueError(f"Client {partition_id} has 0 samples. Try larger alpha or fewer clients.")

    # 80/20 train/val per client
    rng = np.random.RandomState(100 + partition_id)
    idx = rng.permutation(len(Xc))
    split = max(1, int(0.8 * n))

    X_train, y_train = Xc[idx[:split]], yc[idx[:split]]
    X_val, y_val = Xc[idx[split:]], yc[idx[split:]]

    # nếu val rỗng (n=1 hoặc n nhỏ), fallback: dùng train làm val
    if len(X_val) == 0:
        X_val, y_val = X_train, y_train


    train_ds = CSVDataset(X_train, y_train)
    val_ds = CSVDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


# ============================================================
# 6. Train & Eval functions
# ============================================================

def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights_torch.to(device))
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
    """
    Return: accuracy, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted
    y_true/y_pred: 1D int tensors (CPU)
    """
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

    # replace NaN with 0 for safety
    precision = torch.nan_to_num(precision, nan=0.0)
    recall = torch.nan_to_num(recall, nan=0.0)
    f1 = torch.nan_to_num(f1, nan=0.0)

    # Accuracy
    accuracy = tp.sum() / torch.clamp(cm.sum().to(torch.float32), min=1.0)

    # Macro
    precision_macro = precision.mean().item()
    recall_macro = recall.mean().item()
    f1_macro = f1.mean().item()

    # Weighted
    weights = support / torch.clamp(support.sum(), min=1.0)
    precision_weighted = (precision * weights).sum().item()
    recall_weighted = (recall * weights).sum().item()
    f1_weighted = (f1 * weights).sum().item()

    return (accuracy.item(),
            precision_macro, recall_macro, f1_macro,
            precision_weighted, recall_weighted, f1_weighted)

def load_centralized_testloader(batch_size: int = 256) -> DataLoader:
    _ensure_data_loaded()
    if _test_features is None:
        raise RuntimeError("test_final.csv not found/loaded")
    ds = CSVDataset(_test_features, _test_labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

def get_class_weight_criterion(device):
    """
    Return CrossEntropyLoss with class weights (if available).
    If you haven't computed class weights yet, it falls back to normal CE loss.
    """
    global _class_weights_torch

    if _class_weights_torch is None:
        return nn.CrossEntropyLoss()

    return nn.CrossEntropyLoss(weight=_class_weights_torch.to(device))


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

            total_loss += loss.item()
            batch_count += 1

            pred = torch.argmax(logits, dim=1)

            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    avg_loss = total_loss / max(batch_count, 1)

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    (acc,
     p_macro, r_macro, f1_macro,
     p_w, r_w, f1_w) = multiclass_metrics(y_true, y_pred, num_classes)

    return avg_loss, acc, p_macro, r_macro, f1_macro, p_w, r_w, f1_w

def show_partition_distribution(
    mode: str,
    num_partitions: int,
    alpha: float = 0.5,
    seed: int = 42,
    top_k_classes: int | None = None,
    plot: bool = True,
):
    """
    Hiển thị phân phối nhãn theo từng client:
    - counts: số mẫu mỗi lớp trên mỗi client
    - ratios: tỉ lệ (%)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    _ensure_data_loaded()
    y = _train_labels
    num_classes = int(y.max()) + 1

    # build/get partitions (indices)
    key = (mode, float(alpha), int(num_partitions))
    if key not in _partitions_cache:
        _partitions_cache[key] = _build_partitions(mode, num_partitions, alpha=alpha, seed=seed)

    parts = _partitions_cache[key]  # list[np.ndarray indices]

    # counts matrix: [num_partitions, num_classes]
    counts = np.zeros((num_partitions, num_classes), dtype=np.int64)
    sizes = np.zeros((num_partitions,), dtype=np.int64)

    for cid in range(num_partitions):
        idxs = np.asarray(parts[cid], dtype=np.int64)
        sizes[cid] = len(idxs)
        if len(idxs) > 0:
            counts[cid] = np.bincount(y[idxs], minlength=num_classes)

    # DataFrame for display
    df_counts = pd.DataFrame(
        counts,
        index=[f"client_{i}" for i in range(num_partitions)],
        columns=[f"class_{j}" for j in range(num_classes)],
    )
    df_counts.insert(0, "num_samples", sizes)

    # ratios
    denom = np.maximum(sizes.reshape(-1, 1), 1)
    ratios = counts / denom
    df_ratios = pd.DataFrame(
        ratios,
        index=df_counts.index,
        columns=df_counts.columns[1:],  # class columns only
    )

    # optional: show only top_k most frequent classes globally (for readability)
    if top_k_classes is not None and top_k_classes < num_classes:
        global_counts = counts.sum(axis=0)
        top_idx = np.argsort(global_counts)[::-1][:top_k_classes]
        keep_cols = ["num_samples"] + [f"class_{j}" for j in top_idx]
        df_counts = df_counts[keep_cols]
        df_ratios = df_ratios[[f"class_{j}" for j in top_idx]]
        counts_to_plot = counts[:, top_idx]
        class_labels = [f"class_{j}" for j in top_idx]
    else:
        counts_to_plot = counts
        class_labels = [f"class_{j}" for j in range(num_classes)]

    # Print summary
    print(f"\n=== Partition distribution | mode={mode} | alpha={alpha} | num_partitions={num_partitions} ===")
    print(df_counts)

    # Basic non-IID indicator: per-client entropy (higher => more mixed)
    eps = 1e-12
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    print("\nEntropy per client (higher => more mixed):")
    for i, e in enumerate(entropy):
        print(f"  client_{i}: {e:.4f}  (n={sizes[i]})")

    if plot:
        # Stacked bar chart (counts)
        plt.figure()
        bottom = np.zeros((num_partitions,), dtype=np.int64)
        for j, lab in enumerate(class_labels):
            plt.bar(np.arange(num_partitions), counts_to_plot[:, j], bottom=bottom, label=lab)
            bottom += counts_to_plot[:, j]
        plt.title(f"Dirichlet partition (mode={mode}, alpha={alpha}) - counts")
        plt.xlabel("Client")
        plt.ylabel("Number of samples")
        plt.xticks(np.arange(num_partitions), [str(i) for i in range(num_partitions)])
        if len(class_labels) <= 15:
            plt.legend()
        plt.tight_layout()
        plt.show()

        # Heatmap (ratios)
        plt.figure()
        ratios_to_plot = counts_to_plot / np.maximum(counts_to_plot.sum(axis=1, keepdims=True), 1)
        plt.imshow(ratios_to_plot, aspect="auto")
        plt.title(f"Dirichlet partition (mode={mode}, alpha={alpha}) - class ratios")
        plt.xlabel("Class")
        plt.ylabel("Client")
        plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90)
        plt.yticks(np.arange(num_partitions), [f"c{i}" for i in range(num_partitions)])
        plt.colorbar(label="ratio")
        plt.tight_layout()
        plt.show()

    return df_counts, df_ratios
