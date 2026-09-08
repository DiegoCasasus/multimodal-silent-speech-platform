from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PARTICIPANTS = ["02", "03", "16", "20"]


def parse_job(text: str) -> dict[str, str]:
    """
    Parse a job specification.

    Accepted formats:
      name|processed_root|model_key
      name=processed_root:model_key
    """
    text = text.strip()
    if "|" in text:
        parts = text.split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid --job format: {text!r}. Expected name|processed_root|model_key")
        name, root, model_key = [p.strip() for p in parts]
    elif "=" in text and ":" in text:
        name, rest = text.split("=", 1)
        root, model_key = rest.rsplit(":", 1)
        name, root, model_key = name.strip(), root.strip(), model_key.strip()
    else:
        raise ValueError(
            f"Invalid --job format: {text!r}. Use name|processed_root|model_key "
            "or name=processed_root:model_key"
        )

    if not name or not root or not model_key:
        raise ValueError(f"Invalid --job with empty field: {text!r}")

    return {"name": name, "processed_root": root, "model_key": model_key}


def load_jobs_from_csv(path: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "processed_root", "model_key"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV job file missing columns: {sorted(missing)}")
        for row in reader:
            jobs.append(
                {
                    "name": str(row["name"]).strip(),
                    "processed_root": str(row["processed_root"]).strip(),
                    "model_key": str(row["model_key"]).strip(),
                }
            )
    return jobs


def run_command(cmd: list[str], log_path: Path | None = None) -> int:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 100)

    if log_path is None:
        result = subprocess.run(cmd)
        return int(result.returncode)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return int(process.wait())


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_participant(value: str) -> str:
    text = str(value).strip()
    text = text.replace("sub-", "")
    return text.zfill(2)


def bundle_path_for(train_dir: Path, model_key: str, target_task: str) -> Path:
    return train_dir / f"{model_key}_{target_task}_decoder_bundle.pt"


def metrics_path_for(train_dir: Path, model_key: str, target_task: str) -> Path:
    return train_dir / f"{model_key}_{target_task}_metrics.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run quick selected-participant evaluations for preprocessing/model screening."
    )

    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Job spec: name|processed_root|model_key. Can be repeated.",
    )
    parser.add_argument(
        "--jobs-csv",
        default=None,
        help="Optional CSV with columns: name,processed_root,model_key.",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=DEFAULT_PARTICIPANTS,
        help="Held-out participants to test. Default: 02 03 16 20.",
    )
    parser.add_argument("--target-task", default="silent")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weighting", default="balanced")
    parser.add_argument("--scheduler", default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--augment-time-mask-prob", type=float, default=0.15)
    parser.add_argument("--augment-channel-drop-prob", type=float, default=0.05)

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval if evaluation.json already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any train/eval command fails.",
    )
    parser.add_argument(
        "--write-example-csv",
        default=None,
        help="Write an example jobs CSV and exit.",
    )

    args = parser.parse_args()

    if args.write_example_csv:
        example_path = Path(args.write_example_csv)
        example_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            ["A_current_imu", "data/processed_sweeps/silent_A_current_v3", "imu_only_v1"],
            ["B_imu_basecorr", "data/processed_sweeps/silent_B_imu_basecorr", "imu_only_v1"],
            ["C_imu_accgyro_basecorr", "data/processed_sweeps/silent_C_imu_accgyro_basecorr", "imu_only_v1"],
            ["D_imu_raw_delta", "data/processed_sweeps/silent_D_imu_raw_delta", "imu_only_v1"],
            ["E_emg_envelope", "data/processed_sweeps/silent_E_emg_envelope", "emg_only_v1"],
            ["F_emg_raw_envelope", "data/processed_sweeps/silent_F_emg_raw_envelope", "emg_only_v1"],
            ["G_eeg_car_basecorr", "data/processed_sweeps/silent_G_eeg_car_basecorr", "eeg_only_v1"],
            ["H_eeg_05_30_car_basecorr", "data/processed_sweeps/silent_H_eeg_05_30_car_basecorr", "eeg_only_v1"],
            ["I_all_improved_full", "data/processed_sweeps/silent_I_all_improved_v1", "eeg_emg_imu_v1"],
            ["I_all_improved_imu", "data/processed_sweeps/silent_I_all_improved_v1", "imu_only_v1"],
            ["I_all_improved_eeg_imu", "data/processed_sweeps/silent_I_all_improved_v1", "eeg_imu_v1"],
        ]
        with open(example_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "processed_root", "model_key"])
            writer.writerows(rows)
        print(f"[done] wrote example CSV: {example_path}")
        return 0

    jobs: list[dict[str, str]] = []
    for job_text in args.job:
        jobs.append(parse_job(job_text))
    if args.jobs_csv is not None:
        jobs.extend(load_jobs_from_csv(Path(args.jobs_csv)))

    if not jobs:
        raise SystemExit(
            "No jobs provided. Use --job name|processed_root|model_key, "
            "or --write-example-csv data/selected_preprocessing_jobs.csv"
        )

    participants = [compact_participant(p) for p in args.participants]
    if args.output_root is None:
        raise SystemExit("--output-root is required unless using --write-example-csv")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()

    for job in jobs:
        name = job["name"]
        processed_root = Path(job["processed_root"])
        model_key = job["model_key"]

        if not processed_root.exists():
            msg = f"Processed root does not exist: {processed_root}"
            print(f"[error] {msg}")
            if args.stop_on_error:
                raise FileNotFoundError(msg)
            continue

        for participant in participants:
            run_tag = f"{name}__{model_key}__sub-{participant}"
            train_dir = output_root / name / model_key / f"sub-{participant}" / "train"
            eval_dir = output_root / name / model_key / f"sub-{participant}" / "eval"
            log_dir = output_root / name / model_key / f"sub-{participant}" / "logs"
            eval_json = eval_dir / "evaluation.json"

            if args.skip_existing and eval_json.exists():
                print(f"[skip-existing] {run_tag}")
                eval_data = read_json(eval_json)
                metrics_path = metrics_path_for(train_dir, model_key, args.target_task)
                train_metrics = read_json(metrics_path) if metrics_path.exists() else {}
                rows.append(
                    {
                        "job_name": name,
                        "processed_root": str(processed_root),
                        "model_key": model_key,
                        "participant": participant,
                        "status": "skipped_existing",
                        "n_trials": eval_data.get("n_trials"),
                        "accuracy": eval_data.get("accuracy"),
                        "balanced_accuracy": eval_data.get("balanced_accuracy"),
                        "macro_f1": eval_data.get("macro_f1"),
                        "best_epoch": train_metrics.get("best_epoch"),
                        "best_val_accuracy": train_metrics.get("best_val_accuracy"),
                        "best_val_balanced_accuracy": train_metrics.get("best_val_balanced_accuracy"),
                        "best_val_macro_f1": train_metrics.get("best_val_macro_f1"),
                        "bundle_path": str(bundle_path_for(train_dir, model_key, args.target_task)),
                        "eval_dir": str(eval_dir),
                    }
                )
                continue

            train_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            train_cmd = [
                sys.executable,
                "code/decoder/train_export_decoder.py",
                "--processed-root", str(processed_root),
                "--target-task", args.target_task,
                "--model-key", model_key,
                "--output-dir", str(train_dir),
                "--epochs", str(args.epochs),
                "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--dropout", str(args.dropout),
                "--label-smoothing", str(args.label_smoothing),
                "--class-weighting", args.class_weighting,
                "--scheduler", args.scheduler,
                "--scheduler-patience", str(args.scheduler_patience),
                "--scheduler-factor", str(args.scheduler_factor),
                "--split-mode", "participant",
                "--val-participant", participant,
                "--device", args.device,
                "--augment-time-mask-prob", str(args.augment_time_mask_prob),
                "--augment-channel-drop-prob", str(args.augment_channel_drop_prob),
            ]

            train_rc = run_command(train_cmd, log_path=log_dir / "train.log")
            if train_rc != 0:
                row = {
                    "job_name": name,
                    "processed_root": str(processed_root),
                    "model_key": model_key,
                    "participant": participant,
                    "status": f"train_failed_{train_rc}",
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_root / "selected_eval_summary_partial.csv", index=False)
                if args.stop_on_error:
                    raise RuntimeError(f"Training failed for {run_tag} with code {train_rc}")
                continue

            bundle_path = bundle_path_for(train_dir, model_key, args.target_task)
            eval_cmd = [
                sys.executable,
                "code/decoder/evaluate_decoder_bundle.py",
                "--bundle-path", str(bundle_path),
                "--processed-root", str(processed_root),
                "--target-task", args.target_task,
                "--participant", participant,
                "--output-dir", str(eval_dir),
                "--device", args.device,
            ]

            eval_rc = run_command(eval_cmd, log_path=log_dir / "eval.log")
            if eval_rc != 0:
                row = {
                    "job_name": name,
                    "processed_root": str(processed_root),
                    "model_key": model_key,
                    "participant": participant,
                    "status": f"eval_failed_{eval_rc}",
                    "bundle_path": str(bundle_path),
                    "eval_dir": str(eval_dir),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_root / "selected_eval_summary_partial.csv", index=False)
                if args.stop_on_error:
                    raise RuntimeError(f"Evaluation failed for {run_tag} with code {eval_rc}")
                continue

            eval_data = read_json(eval_json)
            metrics_path = metrics_path_for(train_dir, model_key, args.target_task)
            train_metrics = read_json(metrics_path) if metrics_path.exists() else {}

            row = {
                "job_name": name,
                "processed_root": str(processed_root),
                "model_key": model_key,
                "participant": participant,
                "status": "ok",
                "n_trials": eval_data.get("n_trials"),
                "accuracy": eval_data.get("accuracy"),
                "balanced_accuracy": eval_data.get("balanced_accuracy"),
                "macro_f1": eval_data.get("macro_f1"),
                "best_epoch": train_metrics.get("best_epoch"),
                "best_val_accuracy": train_metrics.get("best_val_accuracy"),
                "best_val_balanced_accuracy": train_metrics.get("best_val_balanced_accuracy"),
                "best_val_macro_f1": train_metrics.get("best_val_macro_f1"),
                "bundle_path": str(bundle_path),
                "eval_dir": str(eval_dir),
            }
            rows.append(row)

            partial_csv = output_root / "selected_eval_summary_partial.csv"
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            print(f"[partial] wrote: {partial_csv}")

    df = pd.DataFrame(rows)
    summary_csv = output_root / "selected_eval_summary.csv"
    summary_json = output_root / "selected_eval_summary.json"
    pivot_csv = output_root / "selected_eval_pivot_balanced_accuracy.csv"
    group_csv = output_root / "selected_eval_group_summary.csv"

    df.to_csv(summary_csv, index=False)

    ok_df = df[df["status"].astype(str).str.startswith("ok") | (df["status"] == "skipped_existing")].copy()

    if not ok_df.empty:
        group_df = (
            ok_df.groupby(["job_name", "model_key"], as_index=False)
            .agg(
                n_runs=("balanced_accuracy", "count"),
                mean_accuracy=("accuracy", "mean"),
                mean_balanced_accuracy=("balanced_accuracy", "mean"),
                mean_macro_f1=("macro_f1", "mean"),
                std_balanced_accuracy=("balanced_accuracy", "std"),
                min_balanced_accuracy=("balanced_accuracy", "min"),
                max_balanced_accuracy=("balanced_accuracy", "max"),
            )
            .sort_values("mean_balanced_accuracy", ascending=False)
        )
        group_df.to_csv(group_csv, index=False)

        pivot = ok_df.pivot_table(
            index=["job_name", "model_key"],
            columns="participant",
            values="balanced_accuracy",
            aggfunc="first",
        )
        pivot["mean"] = pivot.mean(axis=1)
        pivot["std"] = pivot.drop(columns=["mean"]).std(axis=1)
        pivot = pivot.sort_values("mean", ascending=False)
        pivot.to_csv(pivot_csv)
    else:
        group_df = pd.DataFrame()
        pivot = pd.DataFrame()

    summary_payload = {
        "target_task": args.target_task,
        "participants": participants,
        "n_jobs": len(jobs),
        "n_rows": int(len(df)),
        "elapsed_seconds": float(time.time() - started),
        "summary_csv": str(summary_csv),
        "group_csv": str(group_csv),
        "pivot_csv": str(pivot_csv),
        "rows": rows,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 100)
    print("Selected participant evaluation finished")
    print("=" * 100)
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")
    if not group_df.empty:
        print(f"Wrote: {group_csv}")
        print(f"Wrote: {pivot_csv}")
        print("\nGroup summary:")
        print(group_df.to_string(index=False))
        print("\nPivot balanced accuracy:")
        print(pivot.to_string())
    else:
        print("No successful evaluations to summarize.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
