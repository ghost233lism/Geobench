#!/usr/bin/env python3
"""Inspect text geocode run outputs and summarize cache/sample failure modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a text_geocode run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing samples.jsonl and geocode_cache.json.")
    parser.add_argument("--show-failed", type=int, default=20, help="How many failed queries to print.")
    parser.add_argument("--show-success", type=int, default=10, help="How many successful cache entries to print.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cache_path = run_dir / "geocode_cache.json"
    samples_path = run_dir / "samples.jsonl"

    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")

    cache = load_json(cache_path)
    cache_status = Counter()
    cache_codes = Counter()
    cache_messages = Counter()
    success_examples = []
    failed_entries = []
    for query, payload in cache.items():
        cache_status[payload.get("status", "missing")] += 1
        if payload.get("code") is not None:
            cache_codes[payload["code"]] += 1
        if payload.get("message"):
            cache_messages[payload["message"]] += 1
        if payload.get("status") == "success":
            if len(success_examples) < args.show_success:
                success_examples.append((query, payload))
        else:
            failed_entries.append((query, payload))

    sample_errors = Counter()
    failed_queries = Counter()
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            reward_details = row.get("reward_details", {}) or {}
            err = reward_details.get("error", "none")
            sample_errors[err] += 1
            if err == "pred_geocode_failed":
                failed_queries[row.get("parsed_answer") or ""] += 1

    print("== Cache status ==")
    for key, value in cache_status.most_common():
        print(f"{key}: {value}")
    if cache_codes:
        print("\n== Cache HTTP/status codes ==")
        for key, value in cache_codes.most_common():
            print(f"{key}: {value}")
    if cache_messages:
        print("\n== Cache messages ==")
        for key, value in cache_messages.most_common(20):
            print(f"{value}: {key}")

    print("\n== Sample reward_details.error ==")
    for key, value in sample_errors.most_common():
        print(f"{key}: {value}")

    print("\n== Top failed parsed answers ==")
    for query, count in failed_queries.most_common(args.show_failed):
        print(f"{count}: {query}")
        payload = cache.get(query)
        if payload is not None:
            print(json.dumps(payload, ensure_ascii=False))

    print("\n== Successful cache examples ==")
    for query, payload in success_examples:
        print(query)
        print(json.dumps(payload, ensure_ascii=False))

    stale_proxy_errors = cache_messages.get("Missing dependencies for SOCKS support.", 0)
    if stale_proxy_errors:
        print("\n== Diagnosis ==")
        print(
            f"Detected {stale_proxy_errors} cached SOCKS dependency failures. "
            "These are persistent cache entries, not OpenCage rate-limit responses."
        )
        print("If proxy handling has been fixed, rerun with --retry-error-cache or delete geocode_cache.json before rerunning.")


if __name__ == "__main__":
    main()
