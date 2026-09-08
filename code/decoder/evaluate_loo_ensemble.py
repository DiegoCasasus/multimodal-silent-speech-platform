from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from model_runtime import BundleDecoderRuntime  # noqa: E402


def parse_weights(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def find_bundle_map(loo_root: Path) -> dict[str, Path]:
    csv_path = loo_root / "loo_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing loo_summary.csv: {csv_path}")
    df = pd.read_csv(csv_path)
    if "participant" not in df.columns or "bundle_path" not in df.columns:
        raise ValueError(f"{csv_path} must contain participant and bundle_path columns")
    out = {}
    for _, row in df.iterrows():
        participant = str(row["participant"]).zfill(2)
        path = Path(str(row["bundle_path"]))
        out[participant] = path
    return out


def load_participant_npz(processed_root: Path, participant: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sub_dir = processed_root / f"sub-{participant}"
    if not sub_dir.exists():
        raise FileNotFoundError(f"Missing participant folder: {sub_dir}")
    paths = sorted(sub_dir.glob("**/*_preprocessed.npz"))
    if not paths:
        raise FileNotFoundError(f"No preprocessed NPZ files found for participant {participant} in {sub_dir}")

    arrays: dict[str, list[np.ndarray]] = {"eeg": [], "emg": [], "imu": []}
    labels: list[np.ndarray] = []

    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            labels.append(np.asarray(z["labels"], dtype=np.int64))
            for modality in arrays.keys():
                if modality in z.files:
                    arrays[modality].append(np.asarray(z[modality], dtype=np.float32))

    y = np.concatenate(labels, axis=0)
    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items() if v}
    return merged, y


def match_channel_count(x_ntc: np.ndarray, expected_channels: int) -> np.ndarray:
    actual = int(x_ntc.shape[2])
    if actual == expected_channels:
        return x_ntc
    if actual > expected_channels:
        return x_ntc[:, :, :expected_channels]
    pad = np.zeros((x_ntc.shape[0], x_ntc.shape[1], expected_channels - actual), dtype=x_ntc.dtype)
    return np.concatenate([x_ntc, pad], axis=2)


def normalization_arrays(norm_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(norm_cfg["mean"], dtype=np.float32)
    std = np.asarray(norm_cfg["std"], dtype=np.float32)
    if mean.ndim == 1:
        mean = mean[None, :, None]
    elif mean.ndim == 2:
        mean = mean[None, :, :]
    if std.ndim == 1:
        std = std[None, :, None]
    elif std.ndim == 2:
        std = std[None, :, :]
    std = std.copy()
    std[std < 1e-6] = 1.0
    return mean, std


def model_probabilities(runtime: BundleDecoderRuntime, arrays: dict[str, np.ndarray], batch_size: int = 256) -> np.ndarray:
    required = runtime.required_modalities
    n = None
    prepared: dict[str, np.ndarray] = {}

    for modality in required:
        if modality not in arrays:
            raise ValueError(f"Missing modality {modality} in processed arrays")
        x = np.asarray(arrays[modality], dtype=np.float32)  # N,T,C, already preprocessed/resampled
        if x.ndim != 3:
            raise ValueError(f"{modality}: expected N,T,C array, got {x.shape}")
        expected_channels = int(runtime.input_channels[modality])
        x = match_channel_count(x, expected_channels)
        x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # N,C,T
        mean, std = normalization_arrays(runtime.normalization[modality])
        x = ((x - mean) / std).astype(np.float32)
        prepared[modality] = x
        n = x.shape[0] if n is None else n

    if n is None:
        raise RuntimeError("No modalities prepared")

    all_probs = []
    runtime.model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {
                modality: torch.tensor(prepared[modality][start:end], dtype=torch.float32, device=runtime.device)
                for modality in required
            }
            logits = runtime.model(batch)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(y_true == y_pred))
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = tp + fn
        if support > 0:
            recalls.append(tp / support)
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append((2 * tp) / denom)
        else:
            f1s.append(0.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate probability averaging ensemble across two LOO decoder families.")
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-a-loo-root", required=True)
    parser.add_argument("--model-a-processed-root", required=True)
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--model-b-loo-root", required=True)
    parser.add_argument("--model-b-processed-root", required=True)
    parser.add_argument("--weight-b-list", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_map = find_bundle_map(Path(args.model_a_loo_root))
    b_map = find_bundle_map(Path(args.model_b_loo_root))
    participants = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not participants:
        raise RuntimeError("No overlapping participants between the two LOO roots")

    weights_b = parse_weights(args.weight_b_list)
    rows = []

    for participant in participants:
        print(f"[participant {participant}] loading bundles")
        rt_a = BundleDecoderRuntime.load(a_map[participant], device_arg=args.device)
        rt_b = BundleDecoderRuntime.load(b_map[participant], device_arg=args.device)

        arrays_a, y_a = load_participant_npz(Path(args.model_a_processed_root), participant)
        arrays_b, y_b = load_participant_npz(Path(args.model_b_processed_root), participant)
        if y_a.shape != y_b.shape or not np.array_equal(y_a, y_b):
            raise ValueError(f"Label mismatch for participant {participant}")

        probs_a = model_probabilities(rt_a, arrays_a, batch_size=args.batch_size)
        probs_b = model_probabilities(rt_b, arrays_b, batch_size=args.batch_size)
        if probs_a.shape != probs_b.shape:
            raise ValueError(f"Probability shape mismatch for participant {participant}: {probs_a.shape} vs {probs_b.shape}")

        for wb in weights_b:
            probs = (1.0 - wb) * probs_a + wb * probs_b
            pred = np.argmax(probs, axis=1)
            metrics = classification_metrics(y_a, pred, num_classes=probs.shape[1])
            rows.append(
                {
                    "participant": participant,
                    "weight_b": float(wb),
                    "weight_a": float(1.0 - wb),
                    "model_a_name": args.model_a_name,
                    "model_b_name": args.model_b_name,
                    "n_trials": int(len(y_a)),
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ensemble_participant_results.csv", index=False)

    group = (
        df.groupby("weight_b")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    group.columns = ["_".join([str(x) for x in col if str(x)]) for col in group.columns.to_flat_index()]
    group.to_csv(output_dir / "ensemble_group_summary.csv", index=False)

    best_idx = group["balanced_accuracy_mean"].idxmax()
    best = group.loc[best_idx].to_dict()
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps({"best_by_mean_balanced_accuracy": best, "n_participants": len(participants)}, indent=2),
        encoding="utf-8",
    )

    print("\nEnsemble group summary:")
    print(group.to_string(index=False))
    print(f"\nBest weight_b by mean balanced accuracy: {best['weight_b']} | bacc={best['balanced_accuracy_mean']}")
    print(f"Wrote ensemble outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
