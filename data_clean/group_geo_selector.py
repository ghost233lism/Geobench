#!/usr/bin/env python3
"""Run geo_data_selector inference on grouped candidates and keep top rows inside each group."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


LEVEL_SUFFIXES = {"1", "2", "3"}
RESERVED_GEO_FLAGS = {
    "--data-dir",
    "--annotation-file",
    "--image-field",
    "--output-dir",
    "--keep-count",
    "--export-mode",
    "--selected-root-name",
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Use selection.jsonl only as a source of group base names, collect matching xxx_1/xxx_2/xxx_3 "
            "annotation rows, rerun geo_data_selector inference on those grouped candidates, then keep keep-count "
            "rows inside each group."
        )
    )
    parser.add_argument(
        "--selection-jsonl",
        "--input-jsonl",
        dest="selection_jsonl",
        required=True,
        help="Selection JSONL that provides the base image names used to locate grouped candidates.",
    )
    parser.add_argument(
        "--annotation-jsonl",
        "--metadata-jsonl",
        dest="annotation_jsonl",
        required=True,
        help="Annotation JSONL containing grouped images such as xxx_1/xxx_2/xxx_3.",
    )
    parser.add_argument("--output-dir", required=True, help="Root output directory for the grouped rerun.")
    parser.add_argument("--output-jsonl", help="Final grouped selection JSONL. Defaults to output-dir/group_selection.jsonl.")
    parser.add_argument("--report-json", help="Optional report path. Defaults to output-dir/group_selection_report.json.")
    parser.add_argument("--keep-count", type=int, default=1, help="How many rows to keep inside each group.")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Only use the first N groups from selection.jsonl. Useful for debugging.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["weighted", "topk"],
        default="weighted",
        help="Use weighted sampling like geo_data_selector or deterministic top-k by value inside each group.",
    )
    parser.add_argument("--mid-target", type=float, help="Override grouped m_mu.")
    parser.add_argument("--tau-mu", type=float, default=1.0)
    parser.add_argument("--tau-sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--variance-mode", choices=["exp_std", "percentile"], default="exp_std")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--group-export-mode",
        choices=["symlink", "copy", "manifest_only"],
        default="symlink",
        help="How to materialize the final grouped selection.",
    )
    parser.add_argument(
        "--group-selected-root-name",
        default="group_selected_images",
        help="Subdirectory name under output-dir for the final grouped export.",
    )
    parser.add_argument(
        "--geo-output-subdir",
        default="geo_selector_run",
        help="Subdirectory under output-dir used for the intermediate geo_data_selector run.",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_selected_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only keep rows explicitly marked as selected in selection.jsonl."""
    return [row for row in rows if row.get("selected") is True]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def path_stem(path_str: str) -> str:
    return Path(path_str).name.rsplit(".", 1)[0]


def split_group_stem(stem: str) -> Tuple[str, str]:
    if "_" not in stem:
        return stem, stem
    base, suffix = stem.rsplit("_", 1)
    if suffix in LEVEL_SUFFIXES:
        return stem, base
    return stem, stem


def infer_row_stem(row: Dict[str, Any], *, selection_mode: bool) -> str:
    for key in ("relative_path", "image_path", "image_name", "filename", "file_name", "filepath"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return path_stem(value)
    item_id = row.get("item_id")
    if isinstance(item_id, str) and item_id:
        return item_id.rsplit("_", 1)[0] if selection_mode and "_" in item_id else item_id
    raise ValueError(f"Cannot infer image stem from row: {row}")


def infer_image_path(row: Dict[str, Any]) -> str:
    for key in ("image_path", "relative_path", "image_name", "filename", "file_name", "filepath"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Cannot infer image path from annotation row: {row}")


def collect_target_groups(selection_rows: Sequence[Dict[str, Any]], max_samples: int | None = None) -> List[str]:
    groups: List[str] = []
    seen: set[str] = set()
    for row in selection_rows:
        stem = infer_row_stem(row, selection_mode=True)
        _, group_base = split_group_stem(stem)
        if group_base in seen:
            continue
        seen.add(group_base)
        groups.append(group_base)
        if max_samples is not None and len(groups) >= max_samples:
            break
    return groups


def prepare_group_candidates(
    selection_rows: Sequence[Dict[str, Any]],
    annotation_rows: Sequence[Dict[str, Any]],
    annotation_base_dir: Path,
    workspace_dir: Path,
    max_samples: int | None,
) -> Tuple[Path, Path, Dict[str, Any], List[Dict[str, Any]]]:
    target_group_list = collect_target_groups(selection_rows, max_samples=max_samples)
    target_groups = set(target_group_list)
    candidate_root = workspace_dir / "candidate_images"
    annotation_out = workspace_dir / "group_candidates.jsonl"

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    candidate_root.mkdir(parents=True, exist_ok=True)

    grouped_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    group_sizes: Dict[str, int] = defaultdict(int)

    for row in annotation_rows:
        image_ref = infer_image_path(row)
        original_path = Path(image_ref).expanduser()
        if not original_path.is_absolute():
            original_path = (annotation_base_dir / original_path).resolve()
        member_stem = path_stem(image_ref)
        _, group_base = split_group_stem(member_stem)
        if group_base not in target_groups:
            continue

        ext = original_path.suffix
        relative_path = Path(group_base) / f"{member_stem}{ext}"
        link_path = candidate_root / relative_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(original_path, link_path)

        new_row = dict(row)
        new_row["image_path"] = str(link_path.resolve())
        new_row["relative_path"] = str(relative_path)
        new_row["source_image_path"] = str(original_path)
        new_row["group_id"] = group_base
        new_row["group_member_stem"] = member_stem
        grouped_candidates[group_base].append(new_row)
        group_sizes[group_base] += 1

    filtered_rows: List[Dict[str, Any]] = []
    singleton_rows: List[Dict[str, Any]] = []
    multi_item_groups = 0
    singleton_groups = 0
    for group_id, rows in grouped_candidates.items():
        if len(rows) == 1:
            singleton_groups += 1
            row = dict(rows[0])
            row["image_path"] = row["source_image_path"]
            row["relative_path"] = str(Path(group_id) / Path(row["source_image_path"]).name)
            row["group_size"] = 1
            row["selection_reason"] = "singleton_group_no_inference"
            row["selected"] = True
            singleton_rows.append(row)
            continue
        multi_item_groups += 1
        filtered_rows.extend(rows)

    write_jsonl(annotation_out, filtered_rows)
    stats = {
        "selection_rows": len(selection_rows),
        "selection_groups": len(target_group_list),
        "max_samples": max_samples,
        "annotation_rows": len(annotation_rows),
        "matched_rows": sum(group_sizes.values()),
        "matched_groups": len(group_sizes),
        "multi_item_candidate_rows": len(filtered_rows),
        "multi_item_groups": multi_item_groups,
        "singleton_groups_skipped_inference": singleton_groups,
        "group_sizes": dict(sorted(group_sizes.items())),
    }
    return candidate_root, annotation_out, stats, singleton_rows


def compute_group_probabilities(
    rows: Sequence[Dict[str, Any]],
    mid_target: float | None,
    tau_mu: float,
    tau_sigma: float,
    epsilon: float,
    variance_mode: str,
) -> List[Dict[str, Any]]:
    reward_stds = np.array([float(row.get("reward_std", 0.0)) for row in rows], dtype=np.float64)
    weights = reward_stds + float(epsilon)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        probs = np.full_like(weights, fill_value=1.0 / max(1, len(weights)))
    else:
        probs = weights / weight_sum

    output_rows: List[Dict[str, Any]] = []
    for row, sigma, prob in zip(rows, reward_stds, probs):
        new_row = dict(row)
        new_row["group_reward_mean_center"] = None
        new_row["group_reward_mean_std"] = None
        new_row["group_reward_std_std"] = float(np.std(reward_stds))
        new_row["s_mid"] = None
        new_row["s_var"] = float(sigma)
        new_row["value"] = float(prob)
        new_row["probability"] = float(prob)
        new_row["group_probability_formula"] = "(reward_std + epsilon) / sum(reward_std + epsilon)"
        output_rows.append(new_row)
    return output_rows


def weighted_sample_without_replacement(
    rows: Sequence[Dict[str, Any]],
    keep_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if keep_count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    keys = []
    for row in rows:
        prob = max(float(row["probability"]), 1e-12)
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        gumbel = -math.log(-math.log(u))
        score = math.log(prob) + gumbel
        keys.append((score, row))
    keys.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in keys[:keep_count]]


def topk_select(rows: Sequence[Dict[str, Any]], keep_count: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("reward_std", 0.0)),
            float(row["probability"]),
            str(row.get("relative_path", "")),
            str(row.get("item_id", "")),
        ),
        reverse=True,
    )
    return ranked[:keep_count]


def infer_group_from_relative_path(relative_path: str) -> Tuple[str, str]:
    rel = Path(relative_path)
    member_stem = rel.stem
    _, group_base = split_group_stem(member_stem)
    if rel.parent != Path(".") and rel.parent.name:
        group_base = rel.parent.name
    return group_base, member_stem


def run_geo_selector(
    candidate_root: Path,
    annotation_path: Path,
    geo_output_dir: Path,
    passthrough_args: Sequence[str],
) -> None:
    conflicting = [flag for flag in RESERVED_GEO_FLAGS if flag in passthrough_args]
    if conflicting:
        raise ValueError(
            "These arguments are managed by group_geo_selector.py and should not be passed through again: "
            + ", ".join(sorted(conflicting))
        )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "geo_data_selector.py"),
        "--data-dir",
        str(candidate_root),
        "--annotation-file",
        str(annotation_path),
        "--image-field",
        "image_path",
        "--output-dir",
        str(geo_output_dir),
        "--export-mode",
        "manifest_only",
        "--keep-count",
        "999999999",
        *passthrough_args,
    ]
    subprocess.run(cmd, check=True)


def export_selected(rows: Sequence[Dict[str, Any]], output_dir: Path, export_mode: str, root_name: str) -> None:
    if export_mode == "manifest_only":
        return
    selected_root = output_dir / root_name
    for row in rows:
        src = Path(row["image_path"])
        dst = selected_root / row["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if export_mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)


def group_select(
    image_stats: Sequence[Dict[str, Any]],
    keep_count: int,
    selection_mode: str,
    mid_target: float | None,
    tau_mu: float,
    tau_sigma: float,
    epsilon: float,
    variance_mode: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in image_stats:
        group_id, member_stem = infer_group_from_relative_path(str(row.get("relative_path", "")))
        enriched = dict(row)
        enriched["group_id"] = group_id
        enriched["group_member_stem"] = member_stem
        grouped[group_id].append(enriched)

    output_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    summary = {
        "matched_groups": len(grouped),
        "total_rows": len(image_stats),
        "singleton_groups": 0,
        "multi_item_groups": 0,
        "selected_rows": 0,
    }

    for group_id in sorted(grouped):
        rows = grouped[group_id]
        group_keep_count = min(keep_count, len(rows))
        if len(rows) == 1:
            summary["singleton_groups"] += 1
            row = dict(rows[0])
            row["group_size"] = 1
            row["selection_reason"] = "singleton_group"
            row["selected"] = True
            output_rows.append(row)
            selected_rows.append(row)
            summary["selected_rows"] += 1
            continue

        summary["multi_item_groups"] += 1
        scored_rows = compute_group_probabilities(
            rows,
            mid_target=mid_target,
            tau_mu=tau_mu,
            tau_sigma=tau_sigma,
            epsilon=epsilon,
            variance_mode=variance_mode,
        )
        if selection_mode == "weighted":
            picked = weighted_sample_without_replacement(scored_rows, group_keep_count, seed)
        else:
            picked = topk_select(scored_rows, group_keep_count)
        picked_ids = {str(row["item_id"]) for row in picked}
        summary["selected_rows"] += len(picked)

        for row in scored_rows:
            enriched = dict(row)
            enriched["group_size"] = len(rows)
            enriched["selection_reason"] = f"group_{selection_mode}"
            enriched["selected"] = str(row["item_id"]) in picked_ids
            output_rows.append(enriched)
            if enriched["selected"]:
                selected_rows.append(enriched)

    output_rows.sort(key=lambda row: (str(row.get("group_id", "")), str(row.get("relative_path", ""))))
    selected_rows.sort(key=lambda row: (str(row.get("group_id", "")), str(row.get("relative_path", ""))))
    return output_rows, {"summary": summary, "selected_rows": selected_rows}


def main() -> None:
    args, passthrough_args = parse_args()
    if args.keep_count < 1:
        raise ValueError("--keep-count must be >= 1")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else output_dir / "group_selection.jsonl"
    report_json = Path(args.report_json).resolve() if args.report_json else output_dir / "group_selection_report.json"

    selection_rows = read_jsonl(Path(args.selection_jsonl))
    selected_selection_rows = filter_selected_rows(selection_rows)
    if not selected_selection_rows:
        raise ValueError(
            f"No rows with selected=true found in selection jsonl: {Path(args.selection_jsonl).resolve()}"
        )
    annotation_rows = read_jsonl(Path(args.annotation_jsonl))

    workspace_dir = output_dir / "group_candidates_workspace"
    annotation_jsonl_path = Path(args.annotation_jsonl).resolve()
    candidate_root, annotation_path, prep_stats, singleton_rows = prepare_group_candidates(
        selected_selection_rows,
        annotation_rows,
        annotation_jsonl_path.parent,
        workspace_dir,
        args.max_samples,
    )

    geo_output_dir = output_dir / args.geo_output_subdir
    image_stats: List[Dict[str, Any]] = []
    if prep_stats["multi_item_candidate_rows"] > 0:
        run_geo_selector(candidate_root, annotation_path, geo_output_dir, passthrough_args)

        image_stats_path = geo_output_dir / "image_stats.jsonl"
        if not image_stats_path.exists():
            raise FileNotFoundError(f"Expected geo_data_selector output not found: {image_stats_path}")
        image_stats = read_jsonl(image_stats_path)

    output_rows, group_result = group_select(
        image_stats=image_stats,
        keep_count=args.keep_count,
        selection_mode=args.selection_mode,
        mid_target=args.mid_target,
        tau_mu=args.tau_mu,
        tau_sigma=args.tau_sigma,
        epsilon=args.epsilon,
        variance_mode=args.variance_mode,
        seed=args.random_seed,
    )
    output_rows.extend(singleton_rows)
    group_result["selected_rows"].extend(singleton_rows)
    group_result["summary"]["matched_groups"] += prep_stats["singleton_groups_skipped_inference"]
    group_result["summary"]["total_rows"] += len(singleton_rows)
    group_result["summary"]["singleton_groups"] += prep_stats["singleton_groups_skipped_inference"]
    group_result["summary"]["selected_rows"] += len(singleton_rows)
    output_rows.sort(key=lambda row: (str(row.get("group_id", "")), str(row.get("relative_path", ""))))
    group_result["selected_rows"].sort(key=lambda row: (str(row.get("group_id", "")), str(row.get("relative_path", ""))))
    write_jsonl(output_jsonl, output_rows)
    export_selected(group_result["selected_rows"], output_dir, args.group_export_mode, args.group_selected_root_name)

    group_sizes = [size for size in prep_stats["group_sizes"].values()]
    report = {
        "selection_jsonl": str(Path(args.selection_jsonl).resolve()),
        "annotation_jsonl": str(Path(args.annotation_jsonl).resolve()),
        "output_jsonl": str(output_jsonl),
        "geo_output_dir": str(geo_output_dir),
        "candidate_root": str(candidate_root),
        "generated_annotation_jsonl": str(annotation_path),
        "keep_count": args.keep_count,
        "selection_mode": args.selection_mode,
        "variance_mode": args.variance_mode,
        "group_export_mode": args.group_export_mode,
        "geo_passthrough_args": list(passthrough_args),
        "summary": {
            "selection_rows_total": len(selection_rows),
            "selection_rows_selected_true": len(selected_selection_rows),
            **prep_stats,
            **group_result["summary"],
            "min_group_size": min(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
        },
    }
    write_json(report_json, report)

    print(
        json.dumps(
            {
                "output_jsonl": str(output_jsonl),
                "geo_output_dir": str(geo_output_dir),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
