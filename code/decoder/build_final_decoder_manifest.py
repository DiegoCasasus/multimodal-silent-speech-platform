from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DISPLAY_NAMES = {
    "emg_imu_v1": "Best practical EMG+IMU",
    "emg_imu_gated_v1": "Gated EMG+IMU",
    "eeg_emg_imu_gated_v1": "Full EEG+EMG+IMU gated",
    "eeg_imu_v1": "EEG+IMU",
    "eeg_only_v1": "EEG only",
    "emg_only_v1": "EMG only",
    "imu_only_v1": "IMU only",
}

DESCRIPTIONS = {
    "emg_imu_v1": "EMG envelope + IMU raw/delta using convolutional feature concatenation.",
    "emg_imu_gated_v1": "EMG envelope + IMU raw/delta using gated modality fusion.",
    "eeg_emg_imu_gated_v1": "EEG multiband + EMG envelope + IMU raw/delta using gated modality fusion.",
}


def load_bundle_meta(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "bundle" not in payload:
        raise RuntimeError(f"Invalid bundle, no 'bundle' key: {path}")
    bundle = payload["bundle"]
    model_key = str(bundle.get("model_key", path.stem))
    preprocessing = bundle.get("preprocessing_meta") or bundle.get("preprocessing") or {}

    return {
        "name": path.stem,
        "display_name": DISPLAY_NAMES.get(model_key, path.stem.replace("_", " ").title()),
        "model_key": model_key,
        "target_task": bundle.get("target_task", "silent"),
        "bundle_path": str(path).replace("\\", "/"),
        "required_modalities": list(bundle.get("required_modalities", [])),
        "fusion_mode": bundle.get("fusion_mode"),
        "input_channels": bundle.get("input_channels", {}),
        "target_size": bundle.get("target_size", {}),
        "preprocessing_tag": preprocessing.get("preprocessing_tag", ""),
        "window_start_s": preprocessing.get("window_start_s"),
        "window_end_s": preprocessing.get("window_end_s"),
        "best_val_balanced_accuracy": bundle.get("best_val_balanced_accuracy"),
        "best_val_macro_f1": bundle.get("best_val_macro_f1"),
        "description": DESCRIPTIONS.get(model_key, "Decoder bundle exported from train_export_decoder.py."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build decoder_manifest.json from exported final bundles.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    pattern = "**/*.pt" if args.recursive else "*.pt"
    bundle_paths = sorted(bundle_dir.glob(pattern))
    if not bundle_paths:
        raise FileNotFoundError(f"No .pt bundles found in {bundle_dir} with pattern {pattern}")

    decoders = []
    for path in bundle_paths:
        try:
            decoders.append(load_bundle_meta(path))
        except Exception as exc:
            print(f"[skip] {path}: {exc}")

    manifest = {
        "manifest_version": "silent_decoder_manifest_v1",
        "bundle_dir": str(bundle_dir).replace("\\", "/"),
        "n_decoders": len(decoders),
        "decoders": decoders,
    }

    output = Path(args.output) if args.output else bundle_dir / "decoder_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {output}")
    for d in decoders:
        print(f"- {d['display_name']} | {d['model_key']} | requires={d['required_modalities']} | {d['bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
