from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True, help="Root folder containing decoder model folders")
    parser.add_argument("--output-csv", required=True, help="Output CSV summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    models_root = Path(args.models_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(models_root.rglob("*_metrics.json"))

    rows = []
    for metric_path in metric_files:
        with open(metric_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        bundle = payload.get("bundle", {})
        rows.append(
            {
                "metrics_path": str(metric_path),
                "model_key": bundle.get("model_key", ""),
                "target_task": bundle.get("target_task", ""),
                "required_modalities": "+".join(bundle.get("required_modalities", [])),
                "best_epoch": payload.get("best_epoch", ""),
                "best_val_accuracy": payload.get("best_val_accuracy", ""),
                "best_val_loss": payload.get("best_val_loss", ""),
                "val_participant": bundle.get("val_participant", ""),
                "train_trials": bundle.get("train_trials", ""),
                "val_trials": bundle.get("val_trials", ""),
                "folder": metric_path.parent.name,
            }
        )

    rows.sort(
        key=lambda x: (
            float(x["best_val_accuracy"]) if x["best_val_accuracy"] != "" else -1.0,
            -(float(x["best_val_loss"]) if x["best_val_loss"] != "" else 9999.0),
        ),
        reverse=True,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "model_key",
                "target_task",
                "required_modalities",
                "best_epoch",
                "best_val_accuracy",
                "best_val_loss",
                "val_participant",
                "train_trials",
                "val_trials",
                "metrics_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote summary CSV: {output_csv}")
    print(f"[done] total runs found: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())