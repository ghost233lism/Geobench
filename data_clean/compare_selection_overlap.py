#!/usr/bin/env python3
"""Compare selected item overlaps across multiple run directories."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selection overlaps across multiple run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing selection.jsonl.")
    parser.add_argument("--output-json", help="Optional path to save the overlap report.")
    return parser.parse_args()


def read_selected_ids(run_dir: Path) -> Set[str]:
    selection_path = run_dir / "selection.jsonl"
    selected_ids: Set[str] = set()
    with selection_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("selected"):
                selected_ids.add(str(row["item_id"]))
    return selected_ids


def main() -> None:
    args = parse_args()
    runs: List[Dict[str, object]] = []
    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str).resolve()
        selected_ids = read_selected_ids(run_dir)
        runs.append({
            "run_dir": str(run_dir),
            "label": run_dir.name,
            "selected_ids": selected_ids,
            "num_selected": len(selected_ids),
        })

    pairwise = []
    for left, right in combinations(runs, 2):
        overlap = left["selected_ids"] & right["selected_ids"]
        pairwise.append({
            "left": left["label"],
            "right": right["label"],
            "overlap_count": len(overlap),
            "overlap_rate_vs_left": len(overlap) / left["num_selected"] if left["num_selected"] else 0.0,
            "overlap_rate_vs_right": len(overlap) / right["num_selected"] if right["num_selected"] else 0.0,
        })

    common_all = set.intersection(*(run["selected_ids"] for run in runs)) if runs else set()
    union_all = set.union(*(run["selected_ids"] for run in runs)) if runs else set()
    report = {
        "runs": [
            {
                "label": run["label"],
                "run_dir": run["run_dir"],
                "num_selected": run["num_selected"],
            }
            for run in runs
        ],
        "pairwise_overlap": pairwise,
        "all_common_overlap_count": len(common_all),
        "all_union_count": len(union_all),
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
