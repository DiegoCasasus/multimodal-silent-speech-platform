import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


# -----------------------------
# Loading / basic math
# -----------------------------
def load_template(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    if "template_vector" not in data:
        raise ValueError(f"{npz_path} does not contain 'template_vector'.")

    vec = np.asarray(data["template_vector"], dtype=float).ravel()

    meta = {}
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


def l2_normalize(x: np.ndarray):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x.copy()
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    return float(np.linalg.norm(a - b))


# -----------------------------
# Filename / manifest parsing
# -----------------------------
def parse_brainprint_filename(path_or_name):
    """
    Tries to parse filenames like:
      exp_sub_02_ses_001_bl_calibration.npz
      exp_sub_Diego_ses_1_bl_calibration.npz
      exp_sub_02_ses_001_run_1_bl_calibration.npz
      sub-02_ses-001_run-1_task-brainprint_mode-EEG_model-eeg_only_v1.npz
    """
    p = Path(path_or_name)
    stem = p.stem

    patterns = [
        r"^exp_sub_(?P<subject>.+?)_ses_(?P<session>.+?)_run_(?P<run>.+?)_bl_(?P<block>.+?)$",
        r"^exp_sub_(?P<subject>.+?)_ses_(?P<session>.+?)_bl_(?P<block>.+?)$",
        r"^sub-(?P<subject>.+?)_ses-(?P<session>.+?)(?:_run-(?P<run>.+?))?(?:_task-(?P<task>.+?))?(?:_mode-(?P<mode>.+?))?(?:_model-(?P<model>.+?))?$",
    ]

    parsed = {
        "stem": stem,
        "subject": None,
        "session": None,
        "run": None,
        "block": None,
        "task": None,
        "mode": None,
        "model": None,
    }

    for pattern in patterns:
        m = re.match(pattern, stem)
        if m:
            parsed.update({k: v for k, v in m.groupdict().items() if v is not None})
            return parsed

    return parsed


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_manifest_paths(bank_dir: Path):
    return [
        bank_dir / "bank_manifest.json",
        bank_dir.parent / "bank_manifest.json",
        bank_dir / "manifest.json",
    ]


def _flatten_manifest_entries(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ["entries", "files", "records", "bank_entries", "manifest"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value

    return []


def _extract_manifest_mapping(entry):
    bank_name = None
    for key in ["bank_name", "bank_file", "output_name", "output_file", "npz_name", "npz_file", "file", "path"]:
        value = entry.get(key)
        if isinstance(value, str) and value.lower().endswith(".npz"):
            bank_name = Path(value).name
            break

    if bank_name is None:
        return None, None

    source_ref = None
    for key in ["source_name", "source_file", "source_path", "source_xdf", "source_stem", "original_name", "original_file"]:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            source_ref = value
            break

    parsed = {
        "stem": Path(bank_name).stem,
        "subject": None,
        "session": None,
        "run": None,
        "block": None,
        "task": None,
        "mode": None,
        "model": None,
    }

    if source_ref is not None:
        parsed_from_source = parse_brainprint_filename(source_ref)
        for key, value in parsed_from_source.items():
            if key != "stem" and value is not None:
                parsed[key] = value

    for key in ["subject", "session", "run", "block", "task", "mode", "model"]:
        if entry.get(key) is not None:
            parsed[key] = str(entry[key])

    return bank_name, {
        "source_ref": source_ref,
        "parsed": parsed,
        "entry": entry,
    }


def load_bank_manifest_map(bank_dir: Path):
    manifest_map = {}

    for manifest_path in _candidate_manifest_paths(bank_dir):
        if not manifest_path.exists():
            continue

        payload = _read_json(manifest_path)
        entries = _flatten_manifest_entries(payload)
        if not entries:
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            bank_name, info = _extract_manifest_mapping(entry)
            if bank_name is not None and info is not None:
                manifest_map[bank_name] = info

        if manifest_map:
            print(f"[info] loaded manifest mappings from: {manifest_path}")
            return manifest_map

    return manifest_map


# -----------------------------
# Compatibility
# -----------------------------
def values_match(a, b, tol=1e-12):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def are_compatible(rec_a, rec_b, strict_meta_keys):
    if rec_a["vec"].shape != rec_b["vec"].shape:
        return False, {
            "reason": "shape_mismatch",
            "shape_a": list(rec_a["vec"].shape),
            "shape_b": list(rec_b["vec"].shape),
        }

    for key in strict_meta_keys:
        if key in rec_a["meta"] and key in rec_b["meta"]:
            if not values_match(rec_a["meta"][key], rec_b["meta"][key]):
                return False, {
                    "reason": "meta_mismatch",
                    "key": key,
                    "value_a": rec_a["meta"][key],
                    "value_b": rec_b["meta"][key],
                }

    return True, {"reason": "ok"}


# -----------------------------
# Records
# -----------------------------
def load_bank_records(bank_dir: Path):
    bank_paths = sorted(bank_dir.glob("*.npz"))
    if len(bank_paths) == 0:
        raise RuntimeError(f"No .npz files found in bank directory: {bank_dir}")

    manifest_map = load_bank_manifest_map(bank_dir)

    records = []
    for path in bank_paths:
        vec, meta = load_template(path)
        parsed = parse_brainprint_filename(path)

        manifest_info = manifest_map.get(path.name)
        source_ref = None
        if manifest_info is not None:
            source_ref = manifest_info.get("source_ref")
            manifest_parsed = manifest_info.get("parsed", {})
            for key in ["subject", "session", "run", "block", "task", "mode", "model"]:
                if parsed.get(key) is None and manifest_parsed.get(key) is not None:
                    parsed[key] = manifest_parsed[key]

        record = {
            "file": str(path),
            "path": path,
            "name": path.name,
            "stem": path.stem,
            "vec": vec,
            "meta": meta,
            "parsed": parsed,
            "subject": parsed.get("subject"),
            "session": parsed.get("session"),
            "run": parsed.get("run"),
            "block": parsed.get("block"),
            "source_ref": source_ref,
        }
        records.append(record)

    return records


# -----------------------------
# Query-vs-bank mode
# -----------------------------
def compare_query_to_bank(query_path: Path, bank_records, strict_meta_keys):
    query_vec, query_meta = load_template(query_path)
    query_parsed = parse_brainprint_filename(query_path)

    query_rec = {
        "file": str(query_path),
        "path": query_path,
        "name": query_path.name,
        "stem": query_path.stem,
        "vec": query_vec,
        "meta": query_meta,
        "parsed": query_parsed,
        "subject": query_parsed.get("subject"),
        "session": query_parsed.get("session"),
        "run": query_parsed.get("run"),
        "block": query_parsed.get("block"),
    }

    results = []
    for bank_rec in bank_records:
        if bank_rec["path"].resolve() == query_path.resolve():
            continue

        compatible, info = are_compatible(query_rec, bank_rec, strict_meta_keys)
        if not compatible:
            results.append(
                {
                    "file": bank_rec["file"],
                    "status": info["reason"],
                    **{k: v for k, v in info.items() if k != "reason"},
                }
            )
            continue

        cos = cosine_similarity(query_rec["vec"], bank_rec["vec"])
        dist = euclidean_distance(query_rec["vec"], bank_rec["vec"])

        results.append(
            {
                "file": bank_rec["file"],
                "status": "ok",
                "cosine_similarity": cos,
                "euclidean_distance": dist,
                "same_subject": (
                    query_rec["subject"] is not None
                    and bank_rec["subject"] is not None
                    and query_rec["subject"] == bank_rec["subject"]
                ),
                "meta": bank_rec["meta"],
                "parsed": bank_rec["parsed"],
                "source_ref": bank_rec.get("source_ref"),
            }
        )

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["cosine_similarity"], reverse=True)

    return {
        "query_file": str(query_path),
        "query_meta": query_meta,
        "query_parsed": query_parsed,
        "n_compared": len(ok_results),
        "ranked_results": ok_results,
        "skipped_results": [r for r in results if r["status"] != "ok"],
    }


# -----------------------------
# All-vs-all mode
# -----------------------------
def build_pairwise_results(records, strict_meta_keys):
    n = len(records)
    cosine_mat = np.full((n, n), np.nan, dtype=float)
    euclid_mat = np.full((n, n), np.nan, dtype=float)
    compat_mat = np.zeros((n, n), dtype=bool)
    skipped_pairs = []

    for i in range(n):
        cosine_mat[i, i] = 1.0
        euclid_mat[i, i] = 0.0
        compat_mat[i, i] = True

    for i in range(n):
        for j in range(i + 1, n):
            compatible, info = are_compatible(records[i], records[j], strict_meta_keys)
            if not compatible:
                skipped_pairs.append(
                    {
                        "file_a": records[i]["file"],
                        "file_b": records[j]["file"],
                        "status": info["reason"],
                        **{k: v for k, v in info.items() if k != "reason"},
                    }
                )
                continue

            cos = cosine_similarity(records[i]["vec"], records[j]["vec"])
            dist = euclidean_distance(records[i]["vec"], records[j]["vec"])

            cosine_mat[i, j] = cosine_mat[j, i] = cos
            euclid_mat[i, j] = euclid_mat[j, i] = dist
            compat_mat[i, j] = compat_mat[j, i] = True

    return cosine_mat, euclid_mat, compat_mat, skipped_pairs


def build_ranked_neighbors(records, cosine_mat, euclid_mat, compat_mat):
    ranked = []

    for i, rec in enumerate(records):
        neighbors = []
        for j, other in enumerate(records):
            if i == j:
                continue
            if not compat_mat[i, j]:
                continue

            neighbors.append(
                {
                    "file": other["file"],
                    "name": other["name"],
                    "subject": other["subject"],
                    "session": other["session"],
                    "block": other["block"],
                    "same_subject": (
                        rec["subject"] is not None
                        and other["subject"] is not None
                        and rec["subject"] == other["subject"]
                    ),
                    "cosine_similarity": float(cosine_mat[i, j]),
                    "euclidean_distance": float(euclid_mat[i, j]),
                    "meta": other["meta"],
                    "parsed": other["parsed"],
                    "source_ref": other.get("source_ref"),
                }
            )

        neighbors.sort(key=lambda r: r["cosine_similarity"], reverse=True)

        ranked.append(
            {
                "query_file": rec["file"],
                "query_name": rec["name"],
                "query_subject": rec["subject"],
                "query_session": rec["session"],
                "query_block": rec["block"],
                "query_meta": rec["meta"],
                "query_parsed": rec["parsed"],
                "query_source_ref": rec.get("source_ref"),
                "ranked_results": neighbors,
            }
        )

    return ranked


def first_same_subject_rank(neighbors):
    for idx, n in enumerate(neighbors, start=1):
        if n["same_subject"]:
            return idx
    return None


def nanmean_or_none(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def build_summary(records, cosine_mat, compat_mat, ranked_neighbors):
    n = len(records)

    within_subject_cos = []
    between_subject_cos = []

    for i in range(n):
        for j in range(i + 1, n):
            if not compat_mat[i, j]:
                continue
            subj_i = records[i]["subject"]
            subj_j = records[j]["subject"]
            if subj_i is not None and subj_j is not None and subj_i == subj_j:
                within_subject_cos.append(float(cosine_mat[i, j]))
            else:
                between_subject_cos.append(float(cosine_mat[i, j]))

    per_file_same_subject_rank = []
    eligible_queries = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    subject_counts = {}
    for rec in records:
        subj = rec["subject"]
        if subj is None:
            continue
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    for row in ranked_neighbors:
        subj = row["query_subject"]
        has_partner = subj is not None and subject_counts.get(subj, 0) > 1
        rank = first_same_subject_rank(row["ranked_results"])

        per_file_same_subject_rank.append(
            {
                "query_file": row["query_file"],
                "query_name": row["query_name"],
                "subject": subj,
                "session": row["query_session"],
                "first_same_subject_rank": rank,
            }
        )

        if has_partner:
            eligible_queries += 1
            if rank is not None and rank <= 1:
                top1_hits += 1
            if rank is not None and rank <= 3:
                top3_hits += 1
            if rank is not None and rank <= 5:
                top5_hits += 1

    subjects = sorted({rec["subject"] for rec in records if rec["subject"] is not None})
    per_subject_summary = []

    for subj in subjects:
        idxs = [i for i, rec in enumerate(records) if rec["subject"] == subj]
        within_vals = []
        between_vals = []

        for ii, i in enumerate(idxs):
            for j in range(n):
                if i == j or not compat_mat[i, j]:
                    continue
                if records[j]["subject"] == subj:
                    if j > i:
                        within_vals.append(float(cosine_mat[i, j]))
                else:
                    between_vals.append(float(cosine_mat[i, j]))

        per_subject_summary.append(
            {
                "subject": subj,
                "n_sessions": len(idxs),
                "mean_within_subject_cosine": nanmean_or_none(within_vals),
                "mean_between_subject_cosine": nanmean_or_none(between_vals),
            }
        )

    summary = {
        "n_files": n,
        "n_subjects": len(subjects),
        "mean_within_subject_cosine": nanmean_or_none(within_subject_cos),
        "mean_between_subject_cosine": nanmean_or_none(between_subject_cos),
        "eligible_same_subject_queries": eligible_queries,
        "top1_same_subject_hit_rate": (top1_hits / eligible_queries) if eligible_queries > 0 else None,
        "top3_same_subject_hit_rate": (top3_hits / eligible_queries) if eligible_queries > 0 else None,
        "top5_same_subject_hit_rate": (top5_hits / eligible_queries) if eligible_queries > 0 else None,
        "per_file_first_same_subject_rank": per_file_same_subject_rank,
        "per_subject_summary": per_subject_summary,
    }

    return summary


# -----------------------------
# Saving helpers
# -----------------------------
def write_matrix_csv(out_path: Path, records, matrix: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["file"] + [rec["name"] for rec in records]
        writer.writerow(header)

        for rec, row in zip(records, matrix):
            formatted = []
            for val in row:
                if np.isnan(val):
                    formatted.append("")
                else:
                    formatted.append(f"{float(val):.10f}")
            writer.writerow([rec["name"]] + formatted)


def write_metadata_csv(out_path: Path, records):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(
        {
            "file",
            "name",
            "stem",
            "subject",
            "session",
            "run",
            "block",
            "source_ref",
            *[k for rec in records for k in rec["meta"].keys()],
            *[f"parsed_{k}" for rec in records for k in rec["parsed"].keys()],
        }
    )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for rec in records:
            row = {
                "file": rec["file"],
                "name": rec["name"],
                "stem": rec["stem"],
                "subject": rec["subject"],
                "session": rec["session"],
                "run": rec["run"],
                "block": rec["block"],
                "source_ref": rec.get("source_ref"),
            }
            row.update(rec["meta"])
            row.update({f"parsed_{k}": v for k, v in rec["parsed"].items()})
            writer.writerow(row)


def try_save_heatmap(out_path: Path, matrix: np.ndarray, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping heatmap.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Templates")
    ax.set_ylabel("Templates")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["query", "all"], default=None)
    parser.add_argument("--query", default=None, help="Path to query .npz brainprint")
    parser.add_argument("--bank-dir", required=True, help="Directory containing .npz brainprints")
    parser.add_argument("--out", default=None, help="Optional JSON output path for query mode")
    parser.add_argument("--out-dir", default=None, help="Output directory for all-vs-all mode")
    parser.add_argument("--top-k", type=int, default=10, help="How many top matches to print")
    parser.add_argument(
        "--strict-meta-keys",
        nargs="*",
        default=["signal_name", "signal_type", "fs", "tmin", "tmax"],
        help="Metadata keys that must match for two templates to be considered compatible",
    )
    parser.add_argument("--save-heatmap", action="store_true", help="Save matrix heatmaps in all-vs-all mode")
    args = parser.parse_args()

    bank_dir = Path(args.bank_dir)
    if not bank_dir.exists():
        raise FileNotFoundError(f"Bank directory not found: {bank_dir}")

    mode = args.mode
    if mode is None:
        mode = "query" if args.query else "all"

    records = load_bank_records(bank_dir)

    if mode == "query":
        if args.query is None:
            raise ValueError("--query is required in query mode")

        query_path = Path(args.query)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")

        result = compare_query_to_bank(query_path, records, args.strict_meta_keys)

        print(f"Query: {result['query_file']}")
        print(f"Compared against {result['n_compared']} compatible templates\n")

        top_k = min(args.top_k, len(result["ranked_results"]))
        for i, r in enumerate(result["ranked_results"][:top_k], start=1):
            same_subj = "  SAME_SUBJ" if r.get("same_subject") else ""
            src = f"  src={Path(r['source_ref']).name}" if r.get("source_ref") else ""
            print(
                f"{i:2d}. {Path(r['file']).name:35s}  "
                f"cosine={r['cosine_similarity']:.6f}  "
                f"euclid={r['euclidean_distance']:.3f}{same_subj}{src}"
            )

        if result["skipped_results"]:
            print("\nSkipped:")
            for r in result["skipped_results"]:
                print(r)

        if args.out is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nSaved results to: {out_path}")

    elif mode == "all":
        out_dir = Path(args.out_dir) if args.out_dir is not None else (bank_dir / "all_vs_all")
        out_dir.mkdir(parents=True, exist_ok=True)

        cosine_mat, euclid_mat, compat_mat, skipped_pairs = build_pairwise_results(records, args.strict_meta_keys)
        ranked_neighbors = build_ranked_neighbors(records, cosine_mat, euclid_mat, compat_mat)
        summary = build_summary(records, cosine_mat, compat_mat, ranked_neighbors)

        write_matrix_csv(out_dir / "cosine_matrix.csv", records, cosine_mat)
        write_matrix_csv(out_dir / "euclidean_matrix.csv", records, euclid_mat)
        write_metadata_csv(out_dir / "metadata_table.csv", records)

        (out_dir / "ranked_neighbors.json").write_text(
            json.dumps(ranked_neighbors, indent=2),
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (out_dir / "skipped_pairs.json").write_text(
            json.dumps(skipped_pairs, indent=2),
            encoding="utf-8",
        )

        if args.save_heatmap:
            try_save_heatmap(out_dir / "cosine_heatmap.png", cosine_mat, "Cosine similarity matrix")
            try_save_heatmap(out_dir / "euclidean_heatmap.png", euclid_mat, "Euclidean distance matrix")

        print(f"Saved all-vs-all outputs to: {out_dir}")
        print(f"n_files: {summary['n_files']}")
        print(f"n_subjects: {summary['n_subjects']}")
        print(f"mean_within_subject_cosine: {summary['mean_within_subject_cosine']}")
        print(f"mean_between_subject_cosine: {summary['mean_between_subject_cosine']}")
        print(f"top1_same_subject_hit_rate: {summary['top1_same_subject_hit_rate']}")
        print(f"top3_same_subject_hit_rate: {summary['top3_same_subject_hit_rate']}")
        print(f"top5_same_subject_hit_rate: {summary['top5_same_subject_hit_rate']}")


if __name__ == "__main__":
    main()