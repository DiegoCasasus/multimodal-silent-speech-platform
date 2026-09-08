from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_MODALITIES = {
    "eeg_only_v1": ["eeg"],
    "emg_only_v1": ["emg"],
    "imu_only_v1": ["imu"],
    "eeg_emg_v1": ["eeg", "emg"],
    "eeg_imu_v1": ["eeg", "imu"],
    "eeg_emg_imu_v1": ["eeg", "emg", "imu"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an EEG-only adaptation factor from a query brainprint and a local bank. "
            "This is meant to modulate the EEG branch contribution at inference time."
        )
    )
    parser.add_argument("--query", required=True, help="Path to the query brainprint NPZ file.")
    parser.add_argument("--bank-dir", required=True, help="Directory containing bank NPZ files.")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_MODALITIES.keys()),
        help="Decoder model key whose modalities should be weighted.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many top matches to average.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help=(
            "Optional all-vs-all summary.json produced by compare_brainprint.py. "
            "If provided, within/between-subject means are used as calibration anchors."
        ),
    )
    parser.add_argument("--alpha-min", type=float, default=0.60, help="Minimum EEG scale.")
    parser.add_argument("--alpha-max", type=float, default=1.40, help="Maximum EEG scale.")
    parser.add_argument(
        "--fallback-low",
        type=float,
        default=None,
        help="Optional manual lower cosine anchor if no summary.json is available.",
    )
    parser.add_argument(
        "--fallback-high",
        type=float,
        default=None,
        help="Optional manual upper cosine anchor if no summary.json is available.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser.parse_args()


def load_template(npz_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    if "template_vector" not in data:
        raise ValueError(f"{npz_path} does not contain template_vector")

    vec = np.asarray(data["template_vector"], dtype=float).ravel()
    meta: dict[str, Any] = {}
    for key in [
        "signal_name",
        "signal_type",
        "fs",
        "n_target_epochs",
        "n_nontarget_epochs",
        "tmin",
        "tmax",
    ]:
        if key in data:
            value = data[key]
            try:
                meta[key] = value.item()
            except Exception:
                meta[key] = value.tolist()
    return vec, meta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x.copy()
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def compatible(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> bool:
    for key in ["signal_name", "signal_type", "fs", "tmin", "tmax"]:
        if key in meta_a and key in meta_b and meta_a[key] != meta_b[key]:
            return False
    return True


def load_bank(query_path: Path, bank_dir: Path) -> list[dict[str, Any]]:
    if not bank_dir.exists():
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")

    rows: list[dict[str, Any]] = []
    for npz_path in sorted(bank_dir.glob("*.npz")):
        if npz_path.resolve() == query_path.resolve():
            continue
        vec, meta = load_template(npz_path)
        rows.append({"path": npz_path, "vec": vec, "meta": meta})

    if not rows:
        raise RuntimeError("No compatible bank files were found.")
    return rows


def derive_anchors_from_summary(summary_json: Path | None) -> tuple[float | None, float | None]:
    if summary_json is None:
        return None, None
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    low = payload.get("mean_between_subject_cosine")
    high = payload.get("mean_within_subject_cosine")
    if low is None or high is None:
        return None, None
    return float(low), float(high)


def derive_anchors_from_scores(scores: list[float]) -> tuple[float, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Cannot derive anchors from empty score list.")

    low = float(np.percentile(arr, 25))
    high = float(np.percentile(arr, 75))
    if high <= low:
        high = low + 1e-6
    return low, high


def cosine_to_alpha(cosine_value: float, low: float, high: float, alpha_min: float, alpha_max: float) -> float:
    if high <= low:
        return float((alpha_min + alpha_max) / 2.0)
    u = (cosine_value - low) / (high - low)
    u = float(np.clip(u, 0.0, 1.0))
    return float(alpha_min + u * (alpha_max - alpha_min))


def build_modality_weights(model_key: str, eeg_alpha: float) -> tuple[dict[str, float], dict[str, float]]:
    modalities = MODEL_MODALITIES[model_key]
    raw_weights = {m: 1.0 for m in modalities}
    if "eeg" in raw_weights:
        raw_weights["eeg"] = float(eeg_alpha)

    total = float(sum(raw_weights.values()))
    normalized = {m: float(v / total) for m, v in raw_weights.items()}
    return raw_weights, normalized


def main() -> int:
    args = parse_args()

    query_path = Path(args.query).resolve()
    bank_dir = Path(args.bank_dir).resolve()
    out_path = Path(args.out).resolve()
    summary_json = Path(args.summary_json).resolve() if args.summary_json else None

    if not query_path.exists():
        raise FileNotFoundError(f"Query brainprint not found: {query_path}")

    query_vec, query_meta = load_template(query_path)
    bank_rows = load_bank(query_path, bank_dir)

    scores: list[dict[str, Any]] = []
    for row in bank_rows:
        if not compatible(query_meta, row["meta"]):
            continue
        cos = cosine_similarity(query_vec, row["vec"])
        scores.append(
            {
                "file": str(row["path"]),
                "name": row["path"].name,
                "cosine_similarity": float(cos),
                "meta": row["meta"],
            }
        )

    if not scores:
        raise RuntimeError("No compatible bank entries remained after metadata checks.")

    scores.sort(key=lambda r: r["cosine_similarity"], reverse=True)
    top_k = min(args.top_k, len(scores))
    top_scores = scores[:top_k]
    topk_mean = float(np.mean([x["cosine_similarity"] for x in top_scores]))
    top1 = float(top_scores[0]["cosine_similarity"])

    low, high = derive_anchors_from_summary(summary_json)
    anchor_source = "summary_json"

    if low is None or high is None:
        if args.fallback_low is not None and args.fallback_high is not None:
            low = float(args.fallback_low)
            high = float(args.fallback_high)
            anchor_source = "manual"
        else:
            low, high = derive_anchors_from_scores([x["cosine_similarity"] for x in scores])
            anchor_source = "bank_percentiles"

    eeg_alpha = cosine_to_alpha(
        cosine_value=topk_mean,
        low=low,
        high=high,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )

    raw_weights, normalized_weights = build_modality_weights(args.model_key, eeg_alpha)

    payload = {
        "query_file": str(query_path),
        "bank_dir": str(bank_dir),
        "model_key": args.model_key,
        "required_modalities": MODEL_MODALITIES[args.model_key],
        "top_k": int(top_k),
        "top1_cosine": top1,
        "topk_mean_cosine": topk_mean,
        "anchor_low": float(low),
        "anchor_high": float(high),
        "anchor_source": anchor_source,
        "eeg_alpha": float(eeg_alpha),
        "raw_branch_weights": raw_weights,
        "normalized_fusion_weights": normalized_weights,
        "top_matches": top_scores,
        "note": (
            "Only the EEG branch is modulated. EMG and IMU remain unchanged. "
            "Use eeg_alpha to scale the EEG encoder output at inference time."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[adapt] model_key={args.model_key}")
    print(f"[adapt] top1_cosine={top1:.4f}")
    print(f"[adapt] top{top_k}_mean_cosine={topk_mean:.4f}")
    print(f"[adapt] anchors=({low:.4f}, {high:.4f}) from {anchor_source}")
    print(f"[adapt] eeg_alpha={eeg_alpha:.4f}")
    print(f"[adapt] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
