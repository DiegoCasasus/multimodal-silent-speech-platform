from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


MODEL_MODALITIES = {
    # Original early-concatenation models
    "eeg_only_v1": ["eeg"],
    "emg_only_v1": ["emg"],
    "imu_only_v1": ["imu"],
    "eeg_emg_v1": ["eeg", "emg"],
    "emg_imu_v1": ["emg", "imu"],
    "eeg_imu_v1": ["eeg", "imu"],
    "eeg_emg_imu_v1": ["eeg", "emg", "imu"],

    # Late-fusion models: one classifier head per modality, globally learned modality weights.
    "eeg_emg_late_v1": ["eeg", "emg"],
    "eeg_imu_late_v1": ["eeg", "imu"],
    "emg_imu_late_v1": ["emg", "imu"],
    "eeg_emg_imu_late_v1": ["eeg", "emg", "imu"],

    # Gated-fusion models: one classifier head per modality, trial-specific learned modality weights.
    "eeg_emg_gated_v1": ["eeg", "emg"],
    "eeg_imu_gated_v1": ["eeg", "imu"],
    "emg_imu_gated_v1": ["emg", "imu"],
    "eeg_emg_imu_gated_v1": ["eeg", "emg", "imu"],
}


MODEL_FUSION_MODES = {
    key: ("late" if key.endswith("_late_v1") else "gated" if key.endswith("_gated_v1") else "concat")
    for key in MODEL_MODALITIES
}


def fusion_mode_for_model(model_key: str) -> str:
    return MODEL_FUSION_MODES.get(model_key, "concat")

LABEL_MAP = {
    "1 (ONE)": 0,
    "2 (TWO)": 1,
    "3 (THREE)": 2,
    "4 (FOUR)": 3,
    "5 (FIVE)": 4,
    "REJOIN": 5,
    "ASSET": 6,
    "ABORT": 7,
    "FORMATION": 8,
    "REFUEL": 9,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True, help="Root folder containing *_preprocessed.npz files")
    parser.add_argument("--target-task", choices=["silent", "imagined"], required=True)
    parser.add_argument(
        "--model-key",
        choices=list(MODEL_MODALITIES.keys()),
        required=True,
        help="Must match GUI selected_model keys",
    )
    parser.add_argument("--output-dir", required=True, help="Where to export trained decoder bundle")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feat-dim", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["random", "participant"],
        default="participant",
        help="Participant split if possible, otherwise random fallback",
    )
    parser.add_argument("--val-participant", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # lightweight augmentation, still online-safe because only used in training
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--augment-time-mask-prob", type=float, default=0.0)
    parser.add_argument("--augment-time-mask-max-frac", type=float, default=0.1)
    parser.add_argument("--augment-channel-drop-prob", type=float, default=0.0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_scalar(value: Any, default: str = "") -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return str(value.flatten()[0])
    return str(value) if value is not None else default


@dataclass
class AugmentConfig:
    noise_std: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_max_frac: float = 0.1
    channel_drop_prob: float = 0.0


class MultiModalDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        labels: np.ndarray,
        augment: bool = False,
        augment_cfg: AugmentConfig | None = None,
    ):
        self.arrays = arrays
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.augment_cfg = augment_cfg or AugmentConfig()

    def __len__(self) -> int:
        return len(self.labels)

    def _augment_array(self, x: np.ndarray) -> np.ndarray:
        # x shape: C, T
        out = x.copy()

        if self.augment_cfg.noise_std > 0:
            out += np.random.normal(0.0, self.augment_cfg.noise_std, size=out.shape).astype(np.float32)

        if self.augment_cfg.time_mask_prob > 0 and np.random.rand() < self.augment_cfg.time_mask_prob:
            t = out.shape[1]
            max_len = max(1, int(round(t * self.augment_cfg.time_mask_max_frac)))
            mask_len = np.random.randint(1, max_len + 1)
            start = np.random.randint(0, max(1, t - mask_len + 1))
            out[:, start:start + mask_len] = 0.0

        if self.augment_cfg.channel_drop_prob > 0:
            for c in range(out.shape[0]):
                if np.random.rand() < self.augment_cfg.channel_drop_prob:
                    out[c, :] = 0.0

        return out

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x = {}
        for modality, arr in self.arrays.items():
            sample = arr[idx]
            if self.augment:
                sample = self._augment_array(sample)
            x[modality] = torch.tensor(sample, dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class ConvEncoder1D(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)
        z = self.dropout(z)
        z = self.proj(z)
        return z


class MultiModalNet(nn.Module):
    def __init__(
        self,
        input_channels: dict[str, int],
        modalities: list[str],
        num_classes: int,
        dropout: float = 0.3,
        feat_dim: int = 64,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        self.modalities = list(modalities)
        self.fusion_mode = str(fusion_mode or "concat").lower()
        if self.fusion_mode not in {"concat", "late", "gated"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.encoders = nn.ModuleDict(
            {
                modality: ConvEncoder1D(
                    in_channels=input_channels[modality],
                    dropout=dropout,
                    feat_dim=feat_dim,
                )
                for modality in self.modalities
            }
        )

        fusion_dim = feat_dim * len(self.modalities)

        if self.fusion_mode == "concat":
            # Original architecture: concatenate all modality features, then classify.
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        else:
            # Late/gated architectures: each modality predicts its own logits.
            # This prevents a weak modality from directly contaminating the feature
            # space of the stronger modalities.
            self.modality_heads = nn.ModuleDict(
                {
                    modality: nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(feat_dim, num_classes),
                    )
                    for modality in self.modalities
                }
            )

            if self.fusion_mode == "late":
                # Global learned reliability weights, shared across trials.
                self.global_logit_weights = nn.Parameter(torch.zeros(len(self.modalities)))
            else:
                # Trial-specific reliability weights predicted from all modality features.
                gate_hidden = max(32, 16 * len(self.modalities))
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim, gate_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden, len(self.modalities)),
                )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        modality_scales: dict[str, float] | None = None,
    ) -> torch.Tensor:
        modality_scales = modality_scales or {}

        feats = []
        for modality in self.modalities:
            feat = self.encoders[modality](batch[modality])
            scale = float(modality_scales.get(modality, 1.0))
            feats.append(feat * scale)

        if self.fusion_mode == "concat":
            fused = torch.cat(feats, dim=1)
            return self.classifier(fused)

        logits_by_modality = [
            self.modality_heads[modality](feat)
            for modality, feat in zip(self.modalities, feats)
        ]
        logits_stack = torch.stack(logits_by_modality, dim=1)  # B, M, C

        if self.fusion_mode == "late":
            weights = torch.softmax(self.global_logit_weights, dim=0).view(1, -1, 1)
        else:
            fused = torch.cat(feats, dim=1)
            weights = torch.softmax(self.gate(fused), dim=1).unsqueeze(-1)  # B, M, 1

        return torch.sum(logits_stack * weights, dim=1)

    def current_fusion_weights(self) -> dict[str, float] | None:
        """Return global weights for late fusion; gated fusion is trial-specific."""
        if self.fusion_mode != "late":
            return None
        with torch.no_grad():
            w = torch.softmax(self.global_logit_weights.detach().cpu(), dim=0).numpy()
        return {m: float(wi) for m, wi in zip(self.modalities, w)}


def compute_channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def stratified_random_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        n_val = min(n_val, len(cls_idx) - 1) if len(cls_idx) > 1 else 1
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)
    return train_idx, val_idx


def participant_split(
    participants: np.ndarray,
    labels: np.ndarray,
    val_participant: str | None,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    unique_parts = sorted(set(participants.tolist()))

    if len(unique_parts) <= 1:
        train_idx, val_idx = stratified_random_split(labels, val_fraction, seed)
        return train_idx, val_idx, None

    if val_participant is not None and val_participant in unique_parts:
        chosen = val_participant
    else:
        chosen = unique_parts[-1]

    val_mask = participants == chosen
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    if len(val_idx) == 0 or len(train_idx) == 0:
        train_idx, val_idx = stratified_random_split(labels, val_fraction, seed)
        return train_idx, val_idx, None

    return train_idx, val_idx, chosen


def discover_npz_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.rglob("*_preprocessed.npz"))


def load_dataset(
    processed_root: Path,
    target_task: str,
    model_key: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    required_modalities = MODEL_MODALITIES[model_key]
    npz_files = discover_npz_files(processed_root)

    arrays_by_modality: dict[str, list[np.ndarray]] = {m: [] for m in required_modalities}
    labels_all: list[np.ndarray] = []
    participants_all: list[np.ndarray] = []
    used_files: list[str] = []
    preprocessing_meta: dict[str, Any] = {}

    expected_shapes: dict[str, tuple[int, int] | None] = {m: None for m in required_modalities}

    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as npz:
            block = safe_scalar(npz.get("block", ""), "")
            if block.lower() != target_task.lower():
                continue

            participant_id = safe_scalar(npz.get("participant_id", ""), "")

            if not preprocessing_meta:
                # Prefer the full preprocessing recipe saved by preprocessing_portable.py.
                if "preprocessing_json" in npz:
                    try:
                        preprocessing_meta = json.loads(safe_scalar(npz.get("preprocessing_json", ""), "{}"))
                    except Exception:
                        preprocessing_meta = {}

                # Backward-compatible fallback for older processed files.
                for key in [
                    "fs",
                    "window_pre_s",
                    "window_post_s",
                    "window_start_s",
                    "window_end_s",
                    "resample_size",
                    "target_size",
                    "cue_delay_s",
                    "notch_hz",
                    "eeg_low_hz",
                    "eeg_high_hz",
                    "emg_low_hz",
                    "emg_high_hz",
                    "imu_lowpass_hz",
                    "preprocessing_version",
                    "preprocessing_tag",
                ]:
                    if key in npz and key not in preprocessing_meta:
                        try:
                            preprocessing_meta[key] = npz[key].item()
                        except Exception:
                            preprocessing_meta[key] = np.asarray(npz[key]).tolist()

            file_arrays: dict[str, np.ndarray] = {}
            missing = False

            for modality in required_modalities:
                if modality not in npz:
                    missing = True
                    break

                arr = np.asarray(npz[modality], dtype=np.float32)  # N, T, C
                if arr.ndim != 3 or arr.shape[0] == 0:
                    missing = True
                    break

                arr = np.transpose(arr, (0, 2, 1))  # N, C, T

                current_shape = (arr.shape[1], arr.shape[2])
                if expected_shapes[modality] is None:
                    expected_shapes[modality] = current_shape
                elif expected_shapes[modality] != current_shape:
                    print(f"[skip] shape mismatch in {npz_path.name} for {modality}: {current_shape} != {expected_shapes[modality]}")
                    missing = True
                    break

                file_arrays[modality] = arr

            if missing:
                continue

            labels = np.asarray(npz["labels"], dtype=np.int64)
            n = len(labels)

            aligned = all(file_arrays[m].shape[0] == n for m in required_modalities)
            if not aligned:
                print(f"[skip] trial count mismatch in {npz_path.name}")
                continue

            for modality in required_modalities:
                arrays_by_modality[modality].append(file_arrays[modality])

            labels_all.append(labels)
            participants_all.append(np.array([participant_id] * n, dtype=object))
            used_files.append(str(npz_path))

    if not labels_all:
        raise RuntimeError("No compatible processed files were found for this task/model combination.")

    final_arrays = {
        modality: np.concatenate(arrays_by_modality[modality], axis=0)
        for modality in required_modalities
    }
    final_labels = np.concatenate(labels_all, axis=0)
    final_participants = np.concatenate(participants_all, axis=0)

    return final_arrays, final_labels, final_participants, used_files, preprocessing_meta


def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict[str, float]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0

    recalls = []
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        denom_recall = tp + fn
        denom_precision = tp + fp

        if denom_recall > 0:
            recall = tp / denom_recall
            recalls.append(float(recall))
        else:
            recall = None

        precision = (tp / denom_precision) if denom_precision > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1s.append(float(2 * precision * recall / (precision + recall)))

    balanced_acc = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += float(loss.item()) * batch_y.size(0)
            total_n += int(batch_y.size(0))

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            labels = batch_y.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    if total_n == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64).tolist(),
        }

    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    cm = compute_confusion_matrix(preds_np, labels_np, num_classes=num_classes)
    derived = metrics_from_confusion_matrix(cm)

    return {
        "loss": total_loss / max(1, total_n),
        "accuracy": derived["accuracy"],
        "balanced_accuracy": derived["balanced_accuracy"],
        "macro_f1": derived["macro_f1"],
        "confusion_matrix": cm.tolist(),
    }


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    processed_root = Path(args.processed_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, labels, participants, used_files, preprocessing_meta = load_dataset(
        processed_root=processed_root,
        target_task=args.target_task,
        model_key=args.model_key,
    )

    if args.split_mode == "participant":
        train_idx, val_idx, chosen_val_participant = participant_split(
            participants=participants,
            labels=labels,
            val_participant=args.val_participant,
            val_fraction=args.val_split,
            seed=args.seed,
        )
    else:
        train_idx, val_idx = stratified_random_split(labels, args.val_split, args.seed)
        chosen_val_participant = None

    train_arrays = {m: arrays[m][train_idx] for m in arrays}
    val_arrays = {m: arrays[m][val_idx] for m in arrays}
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    normalization: dict[str, dict[str, list]] = {}
    for modality in train_arrays:
        mean, std = compute_channel_stats(train_arrays[modality])
        train_arrays[modality] = apply_channel_stats(train_arrays[modality], mean, std)
        val_arrays[modality] = apply_channel_stats(val_arrays[modality], mean, std)
        normalization[modality] = {
            "mean": mean.squeeze(0).tolist(),
            "std": std.squeeze(0).tolist(),
        }

    augment_cfg = AugmentConfig(
        noise_std=args.augment_noise_std,
        time_mask_prob=args.augment_time_mask_prob,
        time_mask_max_frac=args.augment_time_mask_max_frac,
        channel_drop_prob=args.augment_channel_drop_prob,
    )

    train_dataset = MultiModalDataset(train_arrays, train_labels, augment=True, augment_cfg=augment_cfg)
    val_dataset = MultiModalDataset(val_arrays, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_channels = {m: train_arrays[m].shape[1] for m in train_arrays}
    target_size = {m: train_arrays[m].shape[2] for m in train_arrays}
    num_classes = len(LABEL_MAP)

    fusion_mode = fusion_mode_for_model(args.model_key)

    model = MultiModalNet(
        input_channels=input_channels,
        modalities=MODEL_MODALITIES[args.model_key],
        num_classes=num_classes,
        dropout=args.dropout,
        feat_dim=args.feat_dim,
        fusion_mode=fusion_mode,
    ).to(device)

    class_weights_tensor = None
    if args.class_weighting == "balanced":
        class_weights = compute_class_weights(train_labels, num_classes=num_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    else:
        scheduler = None

    history = []
    best_score = -math.inf
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    best_row = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            total_loss += float(loss.item()) * batch_y.size(0)
            total_n += int(batch_y.size(0))

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            labels_batch = batch_y.detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_batch)

        train_preds = np.concatenate(all_preds, axis=0)
        train_targets = np.concatenate(all_labels, axis=0)
        train_cm = compute_confusion_matrix(train_preds, train_targets, num_classes)
        train_metrics = metrics_from_confusion_matrix(train_cm)
        train_metrics["loss"] = total_loss / max(1, total_n)

        val_metrics = evaluate(model, val_loader, device, criterion, num_classes=num_classes)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"train_acc={epoch_row['train_accuracy']:.4f} "
            f"train_bacc={epoch_row['train_balanced_accuracy']:.4f} "
            f"val_loss={epoch_row['val_loss']:.4f} "
            f"val_acc={epoch_row['val_accuracy']:.4f} "
            f"val_bacc={epoch_row['val_balanced_accuracy']:.4f} "
            f"val_f1={epoch_row['val_macro_f1']:.4f}"
        )

        current_score = epoch_row["val_balanced_accuracy"]
        is_better = (
            current_score > best_score
            or (current_score == best_score and epoch_row["val_loss"] < best_val_loss)
        )

        if is_better:
            best_score = current_score
            best_val_loss = epoch_row["val_loss"]
            best_epoch = epoch
            best_row = dict(epoch_row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch >= args.min_epochs and epochs_since_improvement >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epoch(s), stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a best model state.")

    final_row = history[-1]
    final_val_metrics = evaluate(model, val_loader, device, criterion, num_classes=num_classes)

    bundle = {
        "model_key": args.model_key,
        "target_task": args.target_task,
        "required_modalities": MODEL_MODALITIES[args.model_key],
        "fusion_mode": fusion_mode,
        "fusion_weights": model.current_fusion_weights() if hasattr(model, "current_fusion_weights") else None,
        "label_map": LABEL_MAP,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "input_channels": input_channels,
        "target_size": target_size,
        "normalization": normalization,
        "num_classes": num_classes,
        "split_mode": args.split_mode,
        "val_participant": chosen_val_participant,
        "used_files": used_files,
        "train_trials": int(len(train_idx)),
        "val_trials": int(len(val_idx)),
        "train_participants": sorted(set(participants[train_idx].tolist())),
        "val_participants": sorted(set(participants[val_idx].tolist())),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "best_val_balanced_accuracy": float(best_row["val_balanced_accuracy"]),
        "best_val_macro_f1": float(best_row["val_macro_f1"]),
        "best_val_loss": float(best_row["val_loss"]),
        "dropout": float(args.dropout),
        "feat_dim": int(args.feat_dim),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "class_weighting": args.class_weighting,
        "scheduler": args.scheduler,
        "seed": int(args.seed),
        "augment": {
            "noise_std": float(args.augment_noise_std),
            "time_mask_prob": float(args.augment_time_mask_prob),
            "time_mask_max_frac": float(args.augment_time_mask_max_frac),
            "channel_drop_prob": float(args.augment_channel_drop_prob),
        },
        "preprocessing_meta": preprocessing_meta,
        "architecture_name": f"conv1d_{fusion_mode}_fusion_v3",
    }

    model_path = output_dir / f"{args.model_key}_{args.target_task}_decoder_bundle.pt"
    metrics_path = output_dir / f"{args.model_key}_{args.target_task}_metrics.json"

    torch.save(
        {
            "state_dict": best_state,
            "bundle": bundle,
        },
        model_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_val_accuracy": float(best_row["val_accuracy"]),
                "best_val_balanced_accuracy": float(best_row["val_balanced_accuracy"]),
                "best_val_macro_f1": float(best_row["val_macro_f1"]),
                "best_val_loss": float(best_row["val_loss"]),
                "best_row": best_row,
                "final_row": final_row,
                "final_val_metrics": final_val_metrics,
                "history": history,
                "bundle": bundle,
            },
            f,
            indent=2,
        )

    print(f"[done] model exported to: {model_path}")
    print(f"[done] metrics written to: {metrics_path}")
    print(
        f"[best] epoch={best_epoch} "
        f"val_acc={best_row['val_accuracy']:.4f} "
        f"val_bacc={best_row['val_balanced_accuracy']:.4f} "
        f"val_f1={best_row['val_macro_f1']:.4f} "
        f"val_loss={best_row['val_loss']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())