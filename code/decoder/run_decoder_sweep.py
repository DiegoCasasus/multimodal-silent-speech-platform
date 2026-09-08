from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to sweep config JSON")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to use for child runs",
    )
    parser.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parent / "train_export_decoder.py"),
        help="Path to training script",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV summary output path",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop entire sweep on first failed run",
    )
    return parser.parse_args()


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    common = config.get("common", {})
    grid = config.get("grid", {})

    keys = sorted(grid.keys())
    values = [ensure_list(grid[k]) for k in keys]

    experiments = []
    for combo in itertools.product(*values):
        row = dict(common)
        row.update(dict(zip(keys, combo)))
        experiments.append(row)

    return experiments


def build_tag(exp: dict[str, Any]) -> str:
    parts = [
        exp["target_task"],
        exp["model_key"],
        f"seed{exp.get('seed', 0)}",
        f"lr{str(exp.get('learning_rate', 1e-3)).replace('.', 'p')}",
        f"do{str(exp.get('dropout', 0.3)).replace('.', 'p')}",
        f"ls{str(exp.get('label_smoothing', 0.0)).replace('.', 'p')}",
    ]

    processed_root = Path(exp["processed_root"]).name
    parts.insert(0, processed_root)

    return "_".join(parts)


def run_one(
    python_exe: str,
    train_script: str,
    exp: dict[str, Any],
) -> dict[str, Any]:
    output_base = Path(exp["output_root"]).resolve()
    run_tag = build_tag(exp)
    output_dir = output_base / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        train_script,
        "--processed-root", str(exp["processed_root"]),
        "--target-task", str(exp["target_task"]),
        "--model-key", str(exp["model_key"]),
        "--output-dir", str(output_dir),
        "--epochs", str(exp.get("epochs", 60)),
        "--min-epochs", str(exp.get("min_epochs", 15)),
        "--patience", str(exp.get("patience", 12)),
        "--batch-size", str(exp.get("batch_size", 32)),
        "--learning-rate", str(exp.get("learning_rate", 1e-3)),
        "--weight-decay", str(exp.get("weight_decay", 1e-4)),
        "--dropout", str(exp.get("dropout", 0.3)),
        "--feat-dim", str(exp.get("feat_dim", 64)),
        "--label-smoothing", str(exp.get("label_smoothing", 0.0)),
        "--class-weighting", str(exp.get("class_weighting", "balanced")),
        "--scheduler", str(exp.get("scheduler", "plateau")),
        "--scheduler-patience", str(exp.get("scheduler_patience", 5)),
        "--scheduler-factor", str(exp.get("scheduler_factor", 0.5)),
        "--grad-clip", str(exp.get("grad_clip", 1.0)),
        "--seed", str(exp.get("seed", 42)),
        "--val-split", str(exp.get("val_split", 0.2)),
        "--split-mode", str(exp.get("split_mode", "participant")),
        "--device", str(exp.get("device", "cpu")),
        "--augment-noise-std", str(exp.get("augment_noise_std", 0.0)),
        "--augment-time-mask-prob", str(exp.get("augment_time_mask_prob", 0.0)),
        "--augment-time-mask-max-frac", str(exp.get("augment_time_mask_max_frac", 0.1)),
        "--augment-channel-drop-prob", str(exp.get("augment_channel_drop_prob", 0.0)),
    ]

    if exp.get("val_participant") is not None:
        cmd.extend(["--val-participant", str(exp["val_participant"])])

    print("\n[run] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    metrics_path = output_dir / f"{exp['model_key']}_{exp['target_task']}_metrics.json"

    result = {
        "run_tag": run_tag,
        "output_dir": str(output_dir),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        **exp,
    }

    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            result["best_val_accuracy"] = metrics.get("best_val_accuracy")
            result["best_val_balanced_accuracy"] = metrics.get("best_val_balanced_accuracy")
            result["best_val_macro_f1"] = metrics.get("best_val_macro_f1")
            result["best_val_loss"] = metrics.get("best_val_loss")
            result["best_epoch"] = metrics.get("best_epoch")
        except Exception as exc:
            result["metrics_read_error"] = str(exc)

    return result


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for row in rows for k in row.keys() if k not in {"stdout", "stderr"}})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            compact_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(compact_row)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    experiments = expand_experiments(config)
    if not experiments:
        raise RuntimeError("No experiments generated from config.")

    summary_rows = []

    for exp in experiments:
        result = run_one(
            python_exe=args.python_exe,
            train_script=args.train_script,
            exp=exp,
        )
        summary_rows.append(result)

        if result["returncode"] != 0:
            print("[error] run failed")
            print(result["stderr"])
            if args.stop_on_error:
                break

    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else (
        config_path.parent / "decoder_sweep_summary.csv"
    )
    write_summary_csv(summary_csv, summary_rows)

    summary_json = summary_csv.with_suffix(".json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[done] summary csv: {summary_csv}")
    print(f"[done] summary json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())