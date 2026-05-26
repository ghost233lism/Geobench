#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check images referenced by a jsonl metadata file and export unreadable entries."
    )
    parser.add_argument(
        "--annotation-jsonl",
        required=True,
        help="Path to the metadata jsonl file. Each line should contain an image_path field.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV path. Defaults to <annotation-jsonl stem>_bad_images.tsv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads used to load images.",
    )
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Allow PIL to load truncated images instead of flagging them as bad.",
    )
    return parser.parse_args()


def load_records(annotation_jsonl: Path):
    records = []
    with annotation_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "line_no": line_no,
                    "image_path": obj["image_path"],
                    "image_name": obj.get("image_name", ""),
                    "source_id": obj.get("source_id", ""),
                }
            )
    return records


def check_one(record):
    path = Path(record["image_path"])
    if not path.exists():
        return {
            **record,
            "error_type": "missing_file",
            "error_message": "file does not exist",
        }
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as exc:  # noqa: BLE001
        return {
            **record,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return None


def iter_with_progress(records, total):
    use_tqdm = tqdm is not None and (sys.stdout.isatty() or sys.stderr.isatty())
    if not use_tqdm:
        for idx, item in enumerate(records, 1):
            if idx % 1000 == 0 or idx == total:
                print(f"progress: {idx}/{total}", flush=True)
            yield item
        return
    yield from tqdm(
        records,
        total=total,
        desc="Checking images",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )


def write_bad_rows(output_path: Path, bad_rows):
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line_no",
                "image_path",
                "image_name",
                "source_id",
                "error_type",
                "error_message",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(bad_rows)


def main():
    args = parse_args()
    annotation_jsonl = Path(args.annotation_jsonl)
    output_path = (
        Path(args.output)
        if args.output
        else annotation_jsonl.with_name(f"{annotation_jsonl.stem}_bad_images.tsv")
    )

    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated

    records = load_records(annotation_jsonl)
    total = len(records)
    print(f"Loaded {total} records from {annotation_jsonl}")
    print(f"Using {args.workers} workers")

    bad_rows = []
    error_counts = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(check_one, records, chunksize=64)
        for result in iter_with_progress(results, total):
            if result is None:
                continue
            bad_rows.append(result)
            error_counts[result["error_type"]] += 1

    write_bad_rows(output_path, bad_rows)

    print()
    print(f"Finished. Found {len(bad_rows)} bad images.")
    print(f"Output written to: {output_path}")
    if error_counts:
        print("Error summary:")
        for error_type, count in error_counts.most_common():
            print(f"  {error_type}: {count}")
    else:
        print("No bad images found.")


if __name__ == "__main__":
    main()
