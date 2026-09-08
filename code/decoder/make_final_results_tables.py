from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_root_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.name, p
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def load_summary(name: str, root: Path) -> dict[str, Any]:
    json_path = root / "loo_summary.json"
    csv_path = root / "loo_summary.csv"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing summary JSON for {name}: {json_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV for {name}: {csv_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "root": str(root),
        "model_key": data.get("model_key", ""),
        "target_task": data.get("target_task", ""),
        "processed_root": data.get("processed_root", ""),
        "n_participants": int(data.get("n_participants", 0)),
        "mean_accuracy": float(data.get("mean_accuracy", float("nan"))),
        "std_accuracy": float(data.get("std_accuracy", float("nan"))),
        "mean_balanced_accuracy": float(data.get("mean_balanced_accuracy", float("nan"))),
        "std_balanced_accuracy": float(data.get("std_balanced_accuracy", float("nan"))),
        "mean_macro_f1": float(data.get("mean_macro_f1", float("nan"))),
        "std_macro_f1": float(data.get("std_macro_f1", float("nan"))),
    }


def load_participant_table(name: str, root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "loo_summary.csv")
    df["participant"] = df["participant"].astype(str).str.zfill(2)
    keep = ["participant", "accuracy", "balanced_accuracy", "macro_f1"]
    for col in keep:
        if col not in df.columns:
            raise ValueError(f"{root / 'loo_summary.csv'} missing column {col}")
    df = df[keep].copy()
    df["model_name"] = name
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Create final decoder result tables from LOO summaries.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="LOO roots as name=path, e.g. J_emg_imu=data/decoder_loo/silent_J_emg_imu",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [parse_root_arg(x) for x in args.roots]

    summary_rows = [load_summary(name, root) for name, root in roots]
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_balanced_accuracy", ascending=False)
    summary_df.to_csv(output_dir / "final_model_summary.csv", index=False)
    (output_dir / "final_model_summary.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")

    latex_cols = [
        "name",
        "model_key",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
    ]
    latex_text = summary_df[latex_cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    (output_dir / "final_model_summary_latex.txt").write_text(latex_text, encoding="utf-8")

    participant_dfs = [load_participant_table(name, root) for name, root in roots]
    all_participants = pd.concat(participant_dfs, ignore_index=True)
    all_participants.to_csv(output_dir / "final_participant_results_long.csv", index=False)

    bacc_pivot = all_participants.pivot(index="participant", columns="model_name", values="balanced_accuracy")
    f1_pivot = all_participants.pivot(index="participant", columns="model_name", values="macro_f1")
    bacc_pivot.to_csv(output_dir / "participant_balanced_accuracy_matrix.csv")
    f1_pivot.to_csv(output_dir / "participant_macro_f1_matrix.csv")

    best_name = str(summary_df.iloc[0]["name"])
    delta_rows = []
    for name, _root in roots:
        if name == best_name:
            continue
        merged = bacc_pivot[[best_name, name]].dropna().copy()
        merged["delta_best_minus_other"] = merged[best_name] - merged[name]
        delta_rows.append(
            {
                "best_model": best_name,
                "other_model": name,
                "mean_delta_bacc": float(merged["delta_best_minus_other"].mean()),
                "participants_best_higher": int((merged["delta_best_minus_other"] > 0).sum()),
                "participants_other_higher": int((merged["delta_best_minus_other"] < 0).sum()),
                "participants_equal": int((merged["delta_best_minus_other"] == 0).sum()),
            }
        )
    pd.DataFrame(delta_rows).to_csv(output_dir / "pairwise_deltas_vs_best.csv", index=False)

    print("\nFinal model summary:")
    print(summary_df[latex_cols].to_string(index=False))
    print(f"\nWrote result tables to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
