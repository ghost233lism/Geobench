#!/usr/bin/env python3
"""Package selected GeoBench training subsets into per-domain zip archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


GROUP_DOMAINS = {"ground_surface", "map", "remote_sensing"}
PREFERRED_SUFFIXES = (
    "_grpo_no_mean_product",
    "_grpo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package each filtered domain under data_train into a zip containing the selected images "
            "and all json/jsonl artifacts from the chosen selection directory."
        )
    )
    parser.add_argument("--data-train-root", required=True, help="Root directory like .../dataset/data_train")
    parser.add_argument("--output-root", required=True, help="Directory to write per-domain zip files into")
    parser.add_argument(
        "--work-root",
        help="Temporary workspace for assembling per-domain folders. Defaults to <output-root>/_package_work",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        help="Optional subset of domain directory names to package",
    )
    parser.add_argument(
        "--skip-existing-zips",
        action="store_true",
        help="Skip domains whose output zip already exists",
    )
    parser.add_argument(
        "--compression",
        choices=["deflated", "stored"],
        default="deflated",
        help="Zip compression mode. 'stored' is faster but larger.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_primary_run_dir(domain_root: Path) -> Path | None:
    candidates = []
    for child in sorted(domain_root.iterdir()):
        if child.is_dir() and (child / "selection.jsonl").exists():
            candidates.append(child)
    if not candidates:
        return None
    for suffix in PREFERRED_SUFFIXES:
        for candidate in candidates:
            if candidate.name.endswith(suffix):
                return candidate
    return candidates[0]


def pick_domain_source(domain_root: Path) -> tuple[Path, Path] | None:
    if domain_root.name in GROUP_DOMAINS:
        group_dir = next((child for child in sorted(domain_root.iterdir()) if child.is_dir() and (child / "group_selection.jsonl").exists()), None)
        if group_dir is None:
            return None
        return group_dir, group_dir / "group_selection.jsonl"
    run_dir = pick_primary_run_dir(domain_root)
    if run_dir is None:
        return None
    return run_dir, run_dir / "selection.jsonl"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_json_artifacts(source_root: Path, package_root: Path) -> int:
    copied = 0
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            relative = path.relative_to(source_root)
            target = package_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied


def iter_selected_rows(selection_path: Path) -> list[dict]:
    rows = read_jsonl(selection_path)
    selected = [row for row in rows if row.get("selected")]
    return selected if selected else rows


def resolve_image_source(row: dict) -> Path:
    image_path = row.get("image_path")
    if not image_path:
        raise ValueError(f"Missing image_path for row {row.get('item_id')}")
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image path does not exist: {src}")
    return src


def copy_selected_images(selection_path: Path, package_root: Path) -> int:
    image_root = package_root / "image"
    copied = 0
    for row in iter_selected_rows(selection_path):
        src = resolve_image_source(row)
        relative = Path(row.get("relative_path") or src.name)
        target = image_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied += 1
    return copied


def build_zip(source_dir: Path, zip_path: Path, compression: int) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=compression) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(source_dir)))


def package_domain(
    domain_root: Path,
    output_root: Path,
    work_root: Path,
    compression: int,
    skip_existing_zips: bool,
) -> dict | None:
    picked = pick_domain_source(domain_root)
    if picked is None:
        return None
    source_root, selection_path = picked
    zip_path = output_root / f"{domain_root.name}.zip"
    if skip_existing_zips and zip_path.exists():
        return {
            "domain": domain_root.name,
            "source_root": str(source_root),
            "selection_path": str(selection_path),
            "skipped_existing_zip": True,
            "zip_path": str(zip_path),
        }
    package_root = work_root / domain_root.name
    ensure_clean_dir(package_root)
    json_count = copy_json_artifacts(source_root, package_root)
    image_count = copy_selected_images(selection_path, package_root)
    build_zip(package_root, zip_path, compression)
    return {
        "domain": domain_root.name,
        "source_root": str(source_root),
        "selection_path": str(selection_path),
        "json_artifact_count": json_count,
        "selected_image_count": image_count,
        "zip_path": str(zip_path),
    }


def main() -> None:
    args = parse_args()
    data_train_root = Path(args.data_train_root).resolve()
    output_root = Path(args.output_root).resolve()
    work_root = Path(args.work_root).resolve() if args.work_root else output_root / "_package_work"
    compression = ZIP_DEFLATED if args.compression == "deflated" else ZIP_STORED
    domain_filter = set(args.domains or [])
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(work_root)

    reports = []
    for domain_root in sorted(child for child in data_train_root.iterdir() if child.is_dir()):
        if domain_filter and domain_root.name not in domain_filter:
            continue
        report = package_domain(
            domain_root,
            output_root,
            work_root,
            compression=compression,
            skip_existing_zips=args.skip_existing_zips,
        )
        if report is not None:
            reports.append(report)

    summary_path = output_root / "packaging_summary.json"
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"packaged_domains": reports, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
