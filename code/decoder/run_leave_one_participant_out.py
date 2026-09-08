import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def discover_participants(processed_root: Path):
    participants = set()

    for path in processed_root.rglob("*_preprocessed.npz"):
        try:
            data = np.load(path, allow_pickle=True)
            if "participant_id" not in data.files:
                continue

            participant_raw = data["participant_id"]
            participant_str = str(participant_raw)

            # Handles weird stored forms like ['01']
            participant_str = (
                participant_str
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
                .strip()
            )

            participants.add(participant_str)
            data.close()

        except Exception as exc:
            print(f"[warn] could not read {path}: {exc}")

    return sorted(participants)


def run_command(cmd):
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 100)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def load_eval_json(eval_dir: Path):
    evaluation_path = eval_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {evaluation_path}")

    with open(evaluation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run leave-one-participant-out training/evaluation for decoder bundles."
    )

    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--target-task", default="silent")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
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

    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    participants = discover_participants(processed_root)

    if not participants:
        raise RuntimeError(f"No participants found in {processed_root}")

    print(f"[loo] participants: {participants}")

    rows = []

    for participant in participants:
        participant_tag = f"sub-{participant}"
        train_dir = output_root / participant_tag / "train"
        eval_dir = output_root / participant_tag / "eval"

        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "code/decoder/train_export_decoder.py",
            "--processed-root", str(processed_root),
            "--target-task", args.target_task,
            "--model-key", args.model_key,
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

        run_command(train_cmd)

        bundle_path = train_dir / f"{args.model_key}_{args.target_task}_decoder_bundle.pt"

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

        run_command(eval_cmd)

        eval_data = load_eval_json(eval_dir)

        row = {
            "participant": participant,
            "model_key": args.model_key,
            "target_task": args.target_task,
            "n_trials": eval_data.get("n_trials"),
            "accuracy": eval_data.get("accuracy"),
            "balanced_accuracy": eval_data.get("balanced_accuracy"),
            "macro_f1": eval_data.get("macro_f1"),
            "bundle_path": str(bundle_path),
            "eval_dir": str(eval_dir),
        }

        rows.append(row)

        summary_csv = output_root / "loo_summary_partial.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"[loo] partial summary written to: {summary_csv}")

    df = pd.DataFrame(rows)

    summary_csv = output_root / "loo_summary.csv"
    summary_json = output_root / "loo_summary.json"

    df.to_csv(summary_csv, index=False)

    summary = {
        "model_key": args.model_key,
        "target_task": args.target_task,
        "processed_root": str(processed_root),
        "n_participants": len(participants),
        "mean_accuracy": float(df["accuracy"].mean()),
        "std_accuracy": float(df["accuracy"].std()),
        "mean_balanced_accuracy": float(df["balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(df["balanced_accuracy"].std()),
        "mean_macro_f1": float(df["macro_f1"].mean()),
        "std_macro_f1": float(df["macro_f1"].std()),
        "rows": rows,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("[loo] finished")
    print("=" * 100)
    print(df[["participant", "accuracy", "balanced_accuracy", "macro_f1"]])
    print("\nMean accuracy:", summary["mean_accuracy"])
    print("Std accuracy:", summary["std_accuracy"])
    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Std balanced accuracy:", summary["std_balanced_accuracy"])
    print("Mean macro F1:", summary["mean_macro_f1"])
    print("Std macro F1:", summary["std_macro_f1"])
    print(f"\n[loo] wrote: {summary_csv}")
    print(f"[loo] wrote: {summary_json}")


if __name__ == "__main__":
    main()