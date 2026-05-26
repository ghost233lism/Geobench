#!/usr/bin/env python3
"""Post-process an existing geo_data_selector run with a GRPO-oriented scoring rule."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an existing run directory produced by geo_data_selector.py, recompute image values with a "
            "GRPO-oriented score based on reward_mean / reward_std / reward_max, and write a visualization-compatible "
            "run directory for comparison."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing summary.json and image_stats.jsonl.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the post-processed run. It keeps the same file layout as the original run.",
    )
    parser.add_argument(
        "--keep-count",
        type=int,
        help="How many images to keep. Defaults to run_config.keep_count, then summary.num_selected, then all items.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["weighted", "topk"],
        default="weighted",
        help="How to choose the final kept images from the post-processed probabilities.",
    )
    parser.add_argument(
        "--combine-mode",
        choices=["product", "sum"],
        default="product",
        help="How to combine normalized component scores into the final value.",
    )
    parser.add_argument(
        "--include-mean-score",
        action="store_true",
        default=True,
        help="Include mean_score in the combined score.",
    )
    parser.add_argument(
        "--no-include-mean-score",
        dest="include_mean_score",
        action="store_false",
        help="Exclude mean_score from the combined score.",
    )
    parser.add_argument("--mean-weight", type=float, default=0.5, help="Exponent applied to mean_score.")
    parser.add_argument("--std-weight", type=float, default=1.0, help="Exponent applied to std_score.")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Exponent applied to max_score.")
    parser.add_argument(
        "--min-reward-max",
        type=float,
        help="Optional hard filter: rows with reward_max below this threshold get value=0.",
    )
    parser.add_argument(
        "--min-reward-mean",
        type=float,
        help="Optional hard filter: rows with reward_mean below this threshold get value=0.",
    )
    parser.add_argument(
        "--min-reward-std",
        type=float,
        help="Optional hard filter: rows with reward_std below this threshold get value=0.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--export-mode",
        choices=["symlink", "copy", "manifest_only"],
        default="symlink",
        help="How to materialize selected_images in the post-processed run.",
    )
    parser.add_argument(
        "--selected-root-name",
        default="selected_images",
        help="Selected image subdirectory name inside output-dir.",
    )
    return parser.parse_args()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    order = np.argsort(np.argsort(arr, kind="stable"), kind="stable")
    return order.astype(np.float64) / max(1, len(arr) - 1)


def weighted_sample_without_replacement(
    rows: Sequence[Dict[str, Any]],
    keep_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if keep_count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    scored = []
    for row in rows:
        prob = max(float(row["probability"]), 1e-12)
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        gumbel = -math.log(-math.log(u))
        score = math.log(prob) + gumbel
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:keep_count]]


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


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst)


def resolve_keep_count(args: argparse.Namespace, run_config: Dict[str, Any], summary: Dict[str, Any], num_items: int) -> int:
    if args.keep_count is not None:
        return min(args.keep_count, num_items)
    config_keep = run_config.get("keep_count")
    if isinstance(config_keep, int):
        return min(config_keep, num_items)
    summary_keep = summary.get("num_selected")
    if isinstance(summary_keep, int):
        return min(summary_keep, num_items)
    return num_items


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(run_dir / "summary.json", {})
    run_config = read_json(run_dir / "run_config.json", {})
    image_stats = read_jsonl(run_dir / "image_stats.jsonl")
    original_selection = read_jsonl(run_dir / "selection.jsonl")
    if not image_stats:
        raise ValueError(f"No image_stats.jsonl found in {run_dir}")

    original_selected_ids = {row["item_id"] for row in original_selection if row.get("selected")}

    reward_means = np.array([float(row.get("reward_mean", 0.0)) for row in image_stats], dtype=np.float64)
    reward_stds = np.array([float(row.get("reward_std", 0.0)) for row in image_stats], dtype=np.float64)
    reward_maxs = np.array([float(row.get("reward_max", 0.0)) for row in image_stats], dtype=np.float64)

    mean_scores = percentile_ranks(reward_means)
    std_scores = percentile_ranks(reward_stds)
    max_scores = percentile_ranks(reward_maxs)

    processed_rows: List[Dict[str, Any]] = []
    values: List[float] = []
    for idx, row in enumerate(image_stats):
        reward_mean = float(row.get("reward_mean", 0.0))
        reward_std = float(row.get("reward_std", 0.0))
        reward_max = float(row.get("reward_max", 0.0))
        mean_score = float(mean_scores[idx])
        std_score = float(std_scores[idx])
        max_score = float(max_scores[idx])
        passes = True
        if args.min_reward_mean is not None and reward_mean < args.min_reward_mean:
            passes = False
        if args.min_reward_std is not None and reward_std < args.min_reward_std:
            passes = False
        if args.min_reward_max is not None and reward_max < args.min_reward_max:
            passes = False
        active_components: List[tuple[str, float, float]] = []
        if args.include_mean_score:
            active_components.append(("mean_score", mean_score, args.mean_weight))
        active_components.append(("std_score", std_score, args.std_weight))
        active_components.append(("max_score", max_score, args.max_weight))

        if not passes:
            value = 0.0
        elif args.combine_mode == "product":
            value = 1.0
            for _, score, weight in active_components:
                value *= score ** weight
        else:
            total_weight = sum(weight for _, _, weight in active_components)
            weighted_sum = sum(score * weight for _, score, weight in active_components)
            value = weighted_sum / total_weight if total_weight > 0 else 0.0
        new_row = dict(row)
        new_row["original_s_mid"] = row.get("s_mid")
        new_row["original_s_var"] = row.get("s_var")
        new_row["original_value"] = row.get("value")
        new_row["original_probability"] = row.get("probability")
        new_row["original_selected"] = row.get("item_id") in original_selected_ids
        new_row["score_formula"] = (
            "product(" + ", ".join(f"{name}^{weight}" for name, _, weight in active_components) + ")"
            if args.combine_mode == "product"
            else "weighted_average(" + ", ".join(f"{name}*{weight}" for name, _, weight in active_components) + ")"
        )
        new_row["combine_mode"] = args.combine_mode
        new_row["include_mean_score"] = args.include_mean_score
        new_row["score_formula_weights"] = {
            "mean_weight": args.mean_weight,
            "std_weight": args.std_weight,
            "max_weight": args.max_weight,
        }
        new_row["mean_score"] = mean_score
        new_row["std_score"] = std_score
        new_row["max_score"] = max_score
        new_row["score_filter_pass"] = passes
        new_row["s_mid"] = mean_score
        new_row["s_var"] = std_score
        new_row["value"] = float(value)
        values.append(float(value))
        processed_rows.append(new_row)

    values_arr = np.array(values, dtype=np.float64)
    if values_arr.sum() <= 0:
        probs = np.full_like(values_arr, fill_value=1.0 / max(1, len(values_arr)))
    else:
        probs = values_arr / values_arr.sum()
    for row, prob in zip(processed_rows, probs):
        row["probability"] = float(prob)

    keep_count = resolve_keep_count(args, run_config, summary, len(processed_rows))
    if args.selection_mode == "weighted":
        selected_rows = weighted_sample_without_replacement(processed_rows, keep_count, args.random_seed)
    else:
        selected_rows = sorted(
            processed_rows,
            key=lambda row: (float(row["value"]), float(row["probability"]), str(row.get("item_id", ""))),
            reverse=True,
        )[:keep_count]
    selected_ids = {row["item_id"] for row in selected_rows}
    for row in processed_rows:
        row["selected"] = row["item_id"] in selected_ids

    processed_rows.sort(key=lambda row: float(row.get("probability", 0.0)), reverse=True)

    # Reuse heavy artifacts by symlink to keep the comparison output lightweight.
    for name in ("samples.jsonl", "discovered_items.jsonl", "generations"):
        src = run_dir / name
        if src.exists():
            safe_symlink(src, output_dir / name)

    export_selected(selected_rows, output_dir, args.export_mode, args.selected_root_name)
    write_jsonl(output_dir / "image_stats.jsonl", processed_rows)
    write_jsonl(output_dir / "selection.jsonl", processed_rows)

    new_run_config = dict(run_config)
    new_run_config["postprocess_method"] = f"grpo_score_{args.combine_mode}"
    new_run_config["postprocess_source_run_dir"] = str(run_dir)
    new_run_config["postprocess_selection_mode"] = args.selection_mode
    new_run_config["postprocess_combine_mode"] = args.combine_mode
    new_run_config["postprocess_include_mean_score"] = args.include_mean_score
    new_run_config["postprocess_keep_count"] = keep_count
    new_run_config["postprocess_mean_weight"] = args.mean_weight
    new_run_config["postprocess_std_weight"] = args.std_weight
    new_run_config["postprocess_max_weight"] = args.max_weight
    new_run_config["postprocess_min_reward_mean"] = args.min_reward_mean
    new_run_config["postprocess_min_reward_std"] = args.min_reward_std
    new_run_config["postprocess_min_reward_max"] = args.min_reward_max
    new_run_config["postprocess_random_seed"] = args.random_seed
    new_run_config["postprocess_export_mode"] = args.export_mode
    write_json(output_dir / "run_config.json", new_run_config)

    overlap_selected = len(selected_ids & original_selected_ids)
    compare_report = {
        "source_run_dir": str(run_dir),
        "postprocessed_run_dir": str(output_dir),
        "original_num_selected": len(original_selected_ids),
        "postprocessed_num_selected": len(selected_ids),
        "overlap_selected": overlap_selected,
        "overlap_rate_vs_original": overlap_selected / len(original_selected_ids) if original_selected_ids else 0.0,
        "overlap_rate_vs_postprocessed": overlap_selected / len(selected_ids) if selected_ids else 0.0,
        "newly_selected_count": len(selected_ids - original_selected_ids),
        "dropped_selected_count": len(original_selected_ids - selected_ids),
    }
    write_json(output_dir / "compare_report.json", compare_report)

    new_summary = dict(summary)
    new_summary["num_items"] = len(processed_rows)
    new_summary["num_selected"] = len(selected_ids)
    new_summary["export_mode"] = args.export_mode
    new_summary["postprocess_method"] = f"grpo_score_{args.combine_mode}"
    new_summary["postprocess_combine_mode"] = args.combine_mode
    new_summary["postprocess_include_mean_score"] = args.include_mean_score
    new_summary["postprocess_compare_report"] = compare_report
    new_summary["postprocess_weights"] = {
        "mean_weight": args.mean_weight,
        "std_weight": args.std_weight,
        "max_weight": args.max_weight,
    }
    new_summary["postprocess_thresholds"] = {
        "min_reward_mean": args.min_reward_mean,
        "min_reward_std": args.min_reward_std,
        "min_reward_max": args.min_reward_max,
    }
    write_json(output_dir / "summary.json", new_summary)

    print(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "num_items": len(processed_rows),
                "num_selected": len(selected_ids),
                "compare_report": compare_report,
                "visualize_original": f"python /nfs/sunboyuan/Geobench/ms-swift/data_clean/view_sampling_results.py --run-dir {run_dir} --port 8002",
                "visualize_postprocessed": f"python /nfs/sunboyuan/Geobench/ms-swift/data_clean/view_sampling_results.py --run-dir {output_dir} --port 8003",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
