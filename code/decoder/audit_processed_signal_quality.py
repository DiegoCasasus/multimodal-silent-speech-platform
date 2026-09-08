import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_participant_id(value):
    text = str(value)
    text = (
        text.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )
    return text


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_optional_loo_summary(path):
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        print(f"[warn] LOO summary not found: {path}")
        return None

    df = pd.read_csv(path)
    df["participant"] = df["participant"].astype(str).str.zfill(2)
    return df


def modality_stats(arr, pre_fraction=0.5 / 2.5):
    """
    arr expected shape: (trials, time, channels)
    """
    arr = np.asarray(arr, dtype=np.float32)

    n_trials, n_time, n_channels = arr.shape
    finite_mask = np.isfinite(arr)
    nonfinite_count = int((~finite_mask).sum())

    if nonfinite_count > 0:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    flat = arr.reshape(-1, n_channels)
    ch_std = flat.std(axis=0)
    ch_rms = np.sqrt(np.mean(flat ** 2, axis=0))

    global_mean = float(arr.mean())
    global_std = float(arr.std())
    global_abs_mean = float(np.mean(np.abs(arr)))
    global_rms = float(np.sqrt(np.mean(arr ** 2)))
    global_p95_abs = float(np.percentile(np.abs(arr), 95))
    global_max_abs = float(np.max(np.abs(arr)))

    pre_n = max(1, int(round(n_time * pre_fraction)))
    pre_n = min(pre_n, n_time - 1)

    baseline = arr[:, :pre_n, :]
    active = arr[:, pre_n:, :]

    baseline_rms = float(np.sqrt(np.mean(baseline ** 2)))
    active_rms = float(np.sqrt(np.mean(active ** 2)))
    active_to_baseline_rms = active_rms / (baseline_rms + 1e-8)

    # Conservative flat-channel criterion.
    # Absolute threshold catches dead channels; relative threshold catches channels
    # that are almost flat compared with the participant's other channels.
    median_ch_std = float(np.median(ch_std))
    flat_abs = ch_std < 1e-8
    flat_relative = ch_std < (0.01 * median_ch_std + 1e-12)
    flat_channels = int(np.sum(flat_abs | flat_relative))

    return {
        "n_trials": int(n_trials),
        "n_time": int(n_time),
        "n_channels": int(n_channels),
        "nonfinite_count": nonfinite_count,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_abs_mean": global_abs_mean,
        "global_rms": global_rms,
        "global_p95_abs": global_p95_abs,
        "global_max_abs": global_max_abs,
        "baseline_rms": baseline_rms,
        "active_rms": active_rms,
        "active_to_baseline_rms": float(active_to_baseline_rms),
        "channel_std_min": float(np.min(ch_std)),
        "channel_std_median": median_ch_std,
        "channel_std_max": float(np.max(ch_std)),
        "channel_rms_min": float(np.min(ch_rms)),
        "channel_rms_median": float(np.median(ch_rms)),
        "channel_rms_max": float(np.max(ch_rms)),
        "flat_channels": flat_channels,
    }


def add_modality_zscores(df):
    df = df.copy()

    for modality in sorted(df["modality"].unique()):
        mask = df["modality"] == modality

        for col in [
            "global_std",
            "global_abs_mean",
            "global_rms",
            "global_p95_abs",
            "global_max_abs",
            "baseline_rms",
            "active_rms",
            "active_to_baseline_rms",
            "channel_std_median",
            "flat_channels",
        ]:
            values = df.loc[mask, col].astype(float)
            mean = values.mean()
            std = values.std()

            if std == 0 or np.isnan(std):
                df.loc[mask, f"{col}_z"] = 0.0
            else:
                df.loc[mask, f"{col}_z"] = (values - mean) / std

    return df


def add_flags(df):
    df = df.copy()
    flags = []

    for _, row in df.iterrows():
        row_flags = []

        if row["nonfinite_count"] > 0:
            row_flags.append("nonfinite_values")

        if row["flat_channels"] > 0:
            row_flags.append("flat_channels")

        if abs(row.get("global_rms_z", 0.0)) >= 2.0:
            row_flags.append("global_rms_outlier")

        if abs(row.get("global_max_abs_z", 0.0)) >= 2.0:
            row_flags.append("max_abs_outlier")

        if abs(row.get("active_to_baseline_rms_z", 0.0)) >= 2.0:
            row_flags.append("active_baseline_ratio_outlier")

        if row["active_to_baseline_rms"] < 0.9:
            row_flags.append("active_lower_than_baseline")

        flags.append(";".join(row_flags) if row_flags else "")

    df["quality_flags"] = flags
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loo-summary", default=None)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(processed_root.rglob("*_preprocessed.npz"))

    if not npz_files:
        raise RuntimeError(f"No processed NPZ files found in {processed_root}")

    rows = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)

        participant = clean_participant_id(data["participant_id"]) if "participant_id" in data.files else "unknown"
        session = clean_participant_id(data["session_id"]) if "session_id" in data.files else "unknown"

        labels = data["labels"] if "labels" in data.files else None
        n_labels = len(labels) if labels is not None else np.nan
        n_unique_labels = len(set(labels.tolist())) if labels is not None else np.nan

        for modality in ["eeg", "emg", "imu"]:
            if modality not in data.files:
                continue

            arr = data[modality]
            stats = modality_stats(arr)

            row = {
                "participant": str(participant).zfill(2),
                "session": session,
                "file": str(npz_path),
                "modality": modality,
                "n_labels": n_labels,
                "n_unique_labels": n_unique_labels,
            }
            row.update(stats)
            rows.append(row)

        data.close()

    long_df = pd.DataFrame(rows)

    # Participant-level aggregation by modality.
    agg_rules = {
        "n_trials": "sum",
        "n_time": "median",
        "n_channels": "median",
        "n_labels": "sum",
        "n_unique_labels": "median",
        "nonfinite_count": "sum",
        "global_mean": "mean",
        "global_std": "mean",
        "global_abs_mean": "mean",
        "global_rms": "mean",
        "global_p95_abs": "mean",
        "global_max_abs": "max",
        "baseline_rms": "mean",
        "active_rms": "mean",
        "active_to_baseline_rms": "mean",
        "channel_std_min": "min",
        "channel_std_median": "mean",
        "channel_std_max": "max",
        "channel_rms_min": "min",
        "channel_rms_median": "mean",
        "channel_rms_max": "max",
        "flat_channels": "max",
    }

    participant_df = (
        long_df.groupby(["participant", "modality"], as_index=False)
        .agg(agg_rules)
        .sort_values(["participant", "modality"])
    )

    participant_df = add_modality_zscores(participant_df)
    participant_df = add_flags(participant_df)

    loo_df = load_optional_loo_summary(args.loo_summary)

    if loo_df is not None:
        participant_df = participant_df.merge(
            loo_df[["participant", "accuracy", "balanced_accuracy", "macro_f1"]],
            on="participant",
            how="left",
        )

    long_csv = output_dir / "signal_quality_by_file.csv"
    participant_csv = output_dir / "signal_quality_by_participant_modality.csv"
    flags_csv = output_dir / "signal_quality_flags.csv"

    long_df.to_csv(long_csv, index=False)
    participant_df.to_csv(participant_csv, index=False)

    flag_df = participant_df[participant_df["quality_flags"].astype(str) != ""].copy()
    flag_df.to_csv(flags_csv, index=False)

    print("=" * 100)
    print("Signal quality audit finished")
    print("=" * 100)
    print(f"Processed root: {processed_root}")
    print(f"Files read: {len(npz_files)}")
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {participant_csv}")
    print(f"Wrote: {flags_csv}")

    if loo_df is not None:
        print("\nLowest-performing participants from LOO:")
        low = (
            loo_df.sort_values("balanced_accuracy")
            [["participant", "accuracy", "balanced_accuracy", "macro_f1"]]
            .head(8)
        )
        print(low.to_string(index=False))

    print("\nQuality flags:")
    if flag_df.empty:
        print("No major flags detected.")
    else:
        cols = [
            "participant",
            "modality",
            "n_trials",
            "global_rms",
            "active_to_baseline_rms",
            "flat_channels",
            "quality_flags",
        ]
        if "balanced_accuracy" in flag_df.columns:
            cols.insert(2, "balanced_accuracy")

        print(flag_df[cols].to_string(index=False))

    print("\nParticipant-level summary, sorted by participant/modality:")
    display_cols = [
        "participant",
        "modality",
        "n_trials",
        "global_rms",
        "baseline_rms",
        "active_rms",
        "active_to_baseline_rms",
        "flat_channels",
        "quality_flags",
    ]
    if "balanced_accuracy" in participant_df.columns:
        display_cols.insert(2, "balanced_accuracy")

    print(participant_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()