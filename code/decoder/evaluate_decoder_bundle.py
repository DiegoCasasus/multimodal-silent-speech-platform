from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_export_decoder import (
    MultiModalDataset,
    MultiModalNet,
    apply_channel_stats,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-path", required=True, help="Path to exported decoder bundle .pt")
    parser.add_argument("--processed-root", required=True, help="Root folder with *_preprocessed.npz files")
    parser.add_argument("--target-task", choices=["silent", "imagined"], required=True)
    parser.add_argument(
        "--participant",
        default="auto",
        help="Participant to evaluate. Use 'auto' to use bundle['val_participant'], or 'all' for full dataset.",
    )
    parser.add_argument("--output-dir", required=True, help="Folder where evaluation outputs will be written")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def class_metrics_from_confusion(cm: np.ndarray, id_to_label: dict[int, str]) -> list[dict]:
    rows = []
    n_classes = cm.shape[0]

    for cls_idx in range(n_classes):
        tp = int(cm[cls_idx, cls_idx])
        fn = int(cm[cls_idx, :].sum() - tp)
        fp = int(cm[:, cls_idx].sum() - tp)
        support = int(cm[cls_idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class_id": cls_idx,
                "class_label": id_to_label[cls_idx],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return rows


def top_confusions_from_confusion(cm: np.ndarray, id_to_label: dict[int, str], top_k: int = 10) -> list[dict]:
    out = []
    n_classes = cm.shape[0]

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                out.append(
                    {
                        "true_id": i,
                        "true_label": id_to_label[i],
                        "pred_id": j,
                        "pred_label": id_to_label[j],
                        "count": count,
                    }
                )

    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:top_k]


def normalization_to_arrays(norm_block: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(norm_block["mean"], dtype=np.float32)
    std = np.asarray(norm_block["std"], dtype=np.float32)

    if mean.ndim == 1:
        mean = mean[:, None]
    if std.ndim == 1:
        std = std[:, None]

    mean = mean[None, :, :]
    std = std[None, :, :]
    return mean, std


def main() -> int:
    args = parse_args()

    bundle_path = Path(args.bundle_path).resolve()
    processed_root = Path(args.processed_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    checkpoint = torch.load(bundle_path, map_location="cpu")
    bundle = checkpoint["bundle"]
    state_dict = checkpoint["state_dict"]

    model_key = bundle["model_key"]
    required_modalities = bundle["required_modalities"]
    num_classes = int(bundle["num_classes"])

    id_to_label = {int(k): v for k, v in bundle["id_to_label"].items()}

    loaded = load_dataset(
        processed_root=processed_root,
        target_task=args.target_task,
        model_key=model_key,
    )
    if len(loaded) == 5:
        arrays, labels, participants, used_files, preprocessing_meta = loaded
    else:
        arrays, labels, participants, used_files = loaded
        preprocessing_meta = {}

    if args.participant == "auto":
        eval_participant = bundle.get("val_participant")
    elif args.participant == "all":
        eval_participant = None
    else:
        eval_participant = args.participant

    if eval_participant is not None:
        mask = participants == eval_participant
        if not np.any(mask):
            raise RuntimeError(f"No trials found for participant '{eval_participant}'")
        arrays = {m: arrays[m][mask] for m in arrays}
        labels = labels[mask]
        participants = participants[mask]

    for modality in required_modalities:
        mean, std = normalization_to_arrays(bundle["normalization"][modality])
        arrays[modality] = apply_channel_stats(arrays[modality], mean, std)

    dataset = MultiModalDataset(arrays, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    input_channels = {k: int(v) for k, v in bundle["input_channels"].items()}

    model = MultiModalNet(
        input_channels=input_channels,
        modalities=required_modalities,
        num_classes=num_classes,
        dropout=float(bundle.get("dropout", 0.3) or 0.3),
        feat_dim=int(bundle.get("feat_dim", 64) or 64),
        fusion_mode=str(bundle.get("fusion_mode", "concat") or "concat"),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true_all = []
    y_pred_all = []
    conf_all = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_all.extend(batch_y.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())
            conf_all.extend(torch.max(probs, dim=1).values.cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)
    conf = np.asarray(conf_all, dtype=np.float32)

    accuracy = float((y_true == y_pred).mean())

    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    class_rows = class_metrics_from_confusion(cm, id_to_label)

    valid_recalls = [row["recall"] for row in class_rows if row["support"] > 0]
    valid_f1 = [row["f1"] for row in class_rows if row["support"] > 0]

    balanced_accuracy = float(np.mean(valid_recalls)) if valid_recalls else 0.0
    macro_f1 = float(np.mean(valid_f1)) if valid_f1 else 0.0

    top_confusions = top_confusions_from_confusion(cm, id_to_label, top_k=10)

    eval_payload = {
        "bundle_path": str(bundle_path),
        "processed_root": str(processed_root),
        "target_task": args.target_task,
        "model_key": model_key,
        "required_modalities": required_modalities,
        "fusion_mode": str(bundle.get("fusion_mode", "concat") or "concat"),
        "fusion_weights": bundle.get("fusion_weights"),
        "participant": eval_participant if eval_participant is not None else "all",
        "n_trials": int(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mean_confidence": float(np.mean(conf)) if len(conf) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_rows,
        "top_confusions": top_confusions,
        "used_files": used_files,
    }

    json_path = output_dir / "evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    per_trial_csv = output_dir / "trial_predictions.csv"
    with open(per_trial_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_idx", "participant", "true_id", "true_label", "pred_id", "pred_label", "confidence"])
        for i, (t, p, c) in enumerate(zip(y_true, y_pred, conf), start=1):
            writer.writerow(
                [
                    i,
                    participants[i - 1],
                    int(t),
                    id_to_label[int(t)],
                    int(p),
                    id_to_label[int(p)],
                    float(c),
                ]
            )

    class_csv = output_dir / "class_metrics.csv"
    with open(class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_label", "support", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(class_rows)

    print(f"[eval] participant={eval_payload['participant']}")
    print(f"[eval] n_trials={eval_payload['n_trials']}")
    print(f"[eval] accuracy={accuracy:.4f}")
    print(f"[eval] balanced_accuracy={balanced_accuracy:.4f}")
    print(f"[eval] macro_f1={macro_f1:.4f}")
    print(f"[eval] wrote: {json_path}")
    print(f"[eval] wrote: {per_trial_csv}")
    print(f"[eval] wrote: {class_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())