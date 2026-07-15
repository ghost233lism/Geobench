#!/usr/bin/env python3
"""Evaluate GeoVerse-Val with the training GeoScore geocoding API.

This script can:
1. Run ms-swift inference on GeoVerse-Val JSON files with an optional LoRA adapter.
2. Geocode model answers through swift.rewards.orm.GeoScoreAccuracy, the same
   OpenCage backend used during GeoGRPO training.
3. Write enriched JSON outputs and TXT reports in the same metric style as
   GeoVerse-Val/evaluate/eva.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "geobench-val" / "all"
DEFAULT_DATA_VAL_ROOT = SCRIPT_DIR / "geobench-val" / "data_val"
DEFAULT_PROMPT_FILE = REPO_ROOT / "tools" / "system_prompt.txt"
DEFAULT_API_KEY_CONFIG = REPO_ROOT / "tools" / "geoscore_api_keys.conf"
DEFAULT_TRAIN_USER_PROMPT = (
    "<image> Based on the image, tell me the specific location and your thinking process. "
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
)

ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(r'"FinalAnswer"\s*:\s*"((?:\\.|[^"\\])*)"', re.IGNORECASE | re.DOTALL)
DATA_VAL_MARKER = "data_val"
DOMAIN_FIELD = "domain"
ALL_JSONL_DOMAIN_MAP = {
    "gadm_metadata": "GADM_shape",
    "indoor_metadata": "Indoor",
    "vectormap_metadata": "VectorMap",
    "benchmark_1500_ground_surface": "ground_surface",
    "landmark_metadata": "landmark",
    "remote_metadata": "remote_sensing_images",
    "roadnet_metadata": "roadnet-eval",
    "space_metadata": "space",
    "street_metadata": "street",
    "uav_metadata": "uav_eval_500",
}

DISTANCE_THRESHOLDS = {
    "continent_accuracy": 2500.0,
    "country_accuracy": 750.0,
    "region_accuracy": 200.0,
    "city_accuracy": 25.0,
    "street_accuracy": 1.0,
}

METADATA_FIELDS = (
    "continent",
    "continent_code",
    "country",
    "ISO_2",
    "ISO_3",
    "gt_text",
)


class BatchTimeoutError(TimeoutError):
    pass


@dataclass
class InferenceResult:
    rows: List[Dict[str, Any]]
    answer_field: str
    success_field: str
    completed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GeoVerse-Val inference and score with the training GeoScore geocoding API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run inference only.")
    add_infer_args(infer_parser)

    score_parser = subparsers.add_parser("score", help="Geocode and report an existing inference JSON.")
    add_score_args(score_parser)

    report_parser = subparsers.add_parser("report", help="Report an already enriched JSON.")
    add_report_args(report_parser)

    run_parser = subparsers.add_parser("run", help="Run inference, geocode, and report.")
    add_infer_args(run_parser)
    add_score_common_args(run_parser)
    return parser.parse_args()


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="GeoBench validation root. Usually GeoVerse-Val/geobench-val.",
    )
    parser.add_argument(
        "--input-json",
        nargs="*",
        type=Path,
        default=None,
        help="One or more input JSON/JSONL files. If omitted, benchmark JSON files are auto-discovered.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to GeoVerse-Val/eval_runs/<timestamp>.",
    )
    parser.add_argument("--image-field", default="image_path", help="Field containing image paths.")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root for relative image paths. Defaults to each input JSON parent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only process this many rows from each input file. -1 means all rows.",
    )
    parser.add_argument(
        "--no-metadata-patch",
        action="store_true",
        help="Do not backfill continent/country fields from *_metadata.jsonl.",
    )


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    add_common_data_args(parser)
    parser.add_argument(
        "--model-path",
        default="",
        help="Base model path or merged full checkpoint. Required unless it can be inferred from adapter args.json.",
    )
    parser.add_argument(
        "--adapters",
        nargs="*",
        default=[],
        help="Training checkpoint or LoRA adapter path(s), for example output/.../checkpoint-xxx.",
    )
    parser.add_argument("--model-type", default=None, help="Optional ms-swift model_type override.")
    parser.add_argument("--template-type", default=None, help="Optional ms-swift template override.")
    parser.add_argument(
        "--infer-backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend. vllm is the default for evaluation speed.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype for model loading. If omitted, adapter args.json or bfloat16 is used.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--batch-timeout-sec", type=float, default=3000.0, help="Per-batch timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=16384, help="vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=256, help="vLLM max_num_seqs.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size. If omitted, inferred from --gpu.",
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        default='{"image": 1}',
        help="JSON dict passed to vLLM limit_mm_per_prompt.",
    )
    parser.add_argument(
        "--vllm-engine-kwargs",
        default="",
        help="JSON dict of extra keyword arguments passed to ms-swift VllmEngine.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM.")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--user-prompt", default=DEFAULT_TRAIN_USER_PROMPT)
    parser.add_argument("--answer-field", default="", help="Output answer field. Defaults to <run_slug>_answer.")
    parser.add_argument(
        "--success-field",
        default="",
        help="Output inference success field. Defaults to <run_slug>_inference_completed.",
    )
    parser.add_argument(
        "--disable-model-thinking",
        action="store_true",
        help="Build a template with enable_thinking=False when supported by ms-swift.",
    )


def add_score_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latitude-field", default="auto", help="Ground-truth latitude field or auto.")
    parser.add_argument("--longitude-field", default="auto", help="Ground-truth longitude field or auto.")
    parser.add_argument("--continent-field", default="continent_code")
    parser.add_argument(
        "--result-suffix",
        default="_geo",
        help="Suffix appended to answer field for the structured geocode result field.",
    )
    parser.add_argument(
        "--geoscore-api-keys",
        nargs="*",
        default=[],
        help="OpenCage API keys. If omitted, GEOSCORE_API_KEYS or --geoscore-api-key-config is used.",
    )
    parser.add_argument(
        "--geoscore-api-key-config",
        type=Path,
        default=DEFAULT_API_KEY_CONFIG,
        help="Shell-style config file containing GEOSCORE_API_KEYS=...",
    )
    parser.add_argument(
        "--geoscore-cache-file",
        type=Path,
        default=None,
        help="Shared GeoScore cache JSON. Defaults to <output-dir>/geoscore_cache.json.",
    )
    parser.add_argument(
        "--geoscore-max-distance",
        type=float,
        default=2000.0,
        help="Training reward max distance used by GeoScoreAccuracy.",
    )
    parser.add_argument("--geocode-timeout", type=int, default=10, help="OpenCage request timeout.")


def add_score_args(parser: argparse.ArgumentParser) -> None:
    add_common_data_args(parser)
    parser.add_argument("--answer-field", required=True, help="Field containing raw model output.")
    parser.add_argument(
        "--success-field",
        default="",
        help="Boolean inference success field. If omitted, all rows with an answer are treated as successful.",
    )
    add_score_common_args(parser)


def add_report_args(parser: argparse.ArgumentParser) -> None:
    add_common_data_args(parser)
    parser.add_argument("--result-field", required=True, help="Structured result field to summarize.")
    parser.add_argument("--continent-field", default="continent_code")


def ensure_repo_on_path() -> None:
    repo = os.environ.get("SWIFT_REPO_ROOT")
    candidates = [Path(repo)] if repo else []
    candidates.extend([REPO_ROOT, SCRIPT_DIR.parent])
    for candidate in candidates:
        if candidate and candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def split_words(values: Optional[Sequence[str]]) -> List[str]:
    result: List[str] = []
    if not values:
        return result
    for value in values:
        for part in str(value).replace(",", " ").split():
            part = part.strip()
            if part:
                result.append(part)
    return result


def parse_gpu_list(gpu_arg: str) -> List[str]:
    return [item.strip() for item in str(gpu_arg or "").split(",") if item.strip()]


def resolve_tensor_parallel_size(args: argparse.Namespace) -> int:
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor-parallel-size must be >= 1")
        return args.tensor_parallel_size
    return max(1, len(parse_gpu_list(args.gpu)))


def parse_json_dict(text: Optional[str], arg_name: str) -> Optional[Dict[str, Any]]:
    if text is None or str(text).strip() == "":
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be a JSON dict: {text}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{arg_name} must be a JSON dict: {text}")
    return data


def read_api_keys_from_config(path: Path) -> List[str]:
    if not path or not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^\s*GEOSCORE_API_KEYS\s*=\s*['\"]?([^'\"\n]+)", text, flags=re.M)
    if not match:
        return []
    return split_words([match.group(1)])


def resolve_api_keys(args: argparse.Namespace) -> List[str]:
    keys = split_words(getattr(args, "geoscore_api_keys", []))
    if not keys:
        keys = split_words([os.environ.get("GEOSCORE_API_KEYS", "")])
    if not keys:
        keys = read_api_keys_from_config(getattr(args, "geoscore_api_key_config", DEFAULT_API_KEY_CONFIG))
    if keys:
        os.environ["GEOSCORE_API_KEYS"] = ",".join(keys)
    return keys


def default_output_dir() -> Path:
    return SCRIPT_DIR / "eval_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")


def get_data_val_root(data_root: Path) -> Path:
    data_root = data_root.resolve()
    if data_root.name == DATA_VAL_MARKER:
        return data_root
    if (data_root / DATA_VAL_MARKER).exists():
        return data_root / DATA_VAL_MARKER
    return data_root


def discover_input_jsons(data_root: Path) -> List[Path]:
    data_val_root = get_data_val_root(data_root)
    domain_jsonls: List[Path] = []
    for domain_dir in sorted(path for path in data_val_root.iterdir() if path.is_dir()):
        for path in sorted(domain_dir.glob("*_metadata.jsonl")):
            if path.name.lower().startswith("download_"):
                continue
            domain_jsonls.append(path)
    if domain_jsonls:
        return domain_jsonls

    flat_jsonls = [
        path
        for path in sorted(data_val_root.glob("*.jsonl"))
        if not path.name.lower().startswith("download_")
    ]
    if flat_jsonls:
        return flat_jsonls

    candidates = sorted(data_val_root.rglob("*.json")) + sorted(data_val_root.rglob("*.jsonl"))
    result: List[Path] = []
    for path in candidates:
        rel_parts = {part.lower() for part in path.relative_to(data_val_root).parts}
        name = path.name.lower()
        if "shards2" in rel_parts or "metadata" in name or name in {"convert.py", "summary.txt"}:
            continue
        if "bench" not in name and not name.endswith("_500.json") and "remote_" not in name and "indoor_" not in name:
            continue
        result.append(path)
    if not result:
        raise FileNotFoundError(f"No benchmark JSON files found under {data_val_root}")
    return result


def domain_from_path(path: Path, data_val_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(data_val_root.resolve())
    except ValueError:
        return "UNKNOWN"
    if len(relative.parts) == 1:
        return ALL_JSONL_DOMAIN_MAP.get(path.stem.lower(), path.stem)
    return relative.parts[0] if relative.parts else "UNKNOWN"


def annotate_domain(rows: Sequence[Dict[str, Any]], input_json: Path, data_val_root: Path) -> None:
    domain = domain_from_path(input_json, data_val_root)
    if domain == "UNKNOWN":
        return
    for row in rows:
        if row.get(DOMAIN_FIELD) in (None, ""):
            row[DOMAIN_FIELD] = domain


def print_input_summary(input_jsons: Sequence[Path], limit: int) -> None:
    total = 0
    print("Input files:")
    for input_json in input_jsons:
        count = len(load_json_rows(input_json, limit))
        total += count
        print(f"  {count:5d}  {input_json}")
    print(f"Total input samples: {total}")


def resolve_input_jsons(args: argparse.Namespace) -> List[Path]:
    if args.input_json:
        return [path.resolve() for path in args.input_json]
    return discover_input_jsons(args.data_root)


def load_json_rows(path: Path, limit: int = -1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} is not a dict: {path}")
                rows.append(row)
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("data", "samples", "items", "records", "annotations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be a list, or contain a list field: {path}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON item {index} is not a dict: {path}")
        rows.append(item)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")
    return value or "geoverse"


def infer_run_slug(args: argparse.Namespace) -> str:
    if args.answer_field:
        return re.sub(r"_answer$", "", args.answer_field)
    if args.adapters:
        return safe_slug(Path(args.adapters[0]).name)
    if args.model_path:
        return safe_slug(Path(args.model_path).name)
    return "model"


def field_names(args: argparse.Namespace) -> Tuple[str, str]:
    slug = infer_run_slug(args)
    answer_field = args.answer_field or f"{slug}_answer"
    success_field = args.success_field or f"{slug}_inference_completed"
    return answer_field, success_field


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def find_adapter_args(adapters: Sequence[str]) -> Dict[str, Any]:
    for adapter in adapters:
        path = Path(adapter)
        candidates = [path / "args.json"]
        candidates.extend(path.rglob("args.json") if path.exists() else [])
        for candidate in candidates:
            if candidate.exists():
                data = read_json_file(candidate)
                if data:
                    return data
    return {}


def resolve_model_settings(args: argparse.Namespace) -> Tuple[str, Optional[str], Optional[str], str]:
    adapter_args = find_adapter_args(args.adapters)
    model_path = args.model_path or str(
        adapter_args.get("model")
        or adapter_args.get("model_id_or_path")
        or adapter_args.get("model_dir")
        or adapter_args.get("base_model_name_or_path")
        or ""
    )
    if not model_path:
        raise ValueError("--model-path is required when it cannot be inferred from adapter args.json.")

    model_type = args.model_type if args.model_type is not None else adapter_args.get("model_type")
    template_type = args.template_type if args.template_type is not None else adapter_args.get("template")
    torch_dtype = args.torch_dtype or adapter_args.get("torch_dtype") or "bfloat16"
    return model_path, model_type, template_type, str(torch_dtype)


def torch_dtype_from_name(name: str):
    import torch

    normalized = str(name).lower().replace("torch.", "")
    mapping = {
        "auto": None,
        "none": None,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported --torch-dtype: {name}")
    return mapping[normalized]


def build_template_if_needed(
    model_path: str,
    model_type: Optional[str],
    template_type: Optional[str],
    disable_model_thinking: bool,
):
    if not disable_model_thinking:
        return None
    ensure_repo_on_path()
    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(model_id_or_path=model_path, model_type=model_type, download_model=True)
    return get_template(processor, template_type=template_type, enable_thinking=False)


def read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


@contextmanager
def batch_timeout(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):  # type: ignore[unused-argument]
        raise BatchTimeoutError(f"Batch exceeded timeout: {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def infer_batch(engine: Any, requests: List[Any], request_config: Any):
    try:
        return engine.infer(requests, request_config=request_config, use_tqdm=False)
    except TypeError:
        return engine.infer(requests, request_config=request_config)


class MetadataIndex:
    def __init__(self, data_val_root: Path):
        self.data_val_root = data_val_root
        self.by_image_name: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for path in sorted(self.data_val_root.rglob("*metadata.jsonl")):
            with path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(item, dict):
                        continue
                    image_name = item.get("image_name")
                    if image_name:
                        self.by_image_name.setdefault(str(image_name), item)

    def patch_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.load()
        if not self.by_image_name:
            return
        for row in rows:
            image_name = row.get("image_name")
            if not image_name:
                raw_path = row.get("image_path")
                if raw_path:
                    image_name = Path(str(raw_path)).name
            if not image_name:
                continue
            meta = self.by_image_name.get(str(image_name))
            if not meta:
                continue
            for field in METADATA_FIELDS:
                if row.get(field) in (None, "") and meta.get(field) not in (None, ""):
                    row[field] = meta.get(field)


class ImageResolver:
    def __init__(self, data_val_root: Path):
        self.data_val_root = data_val_root
        self._basename_index: Optional[Dict[str, Path]] = None

    def resolve(self, raw_path: Any, image_root: Optional[Path]) -> Optional[str]:
        if raw_path is None:
            return None
        raw = str(raw_path).strip()
        if not raw:
            return None

        path = Path(raw)
        candidates: List[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if image_root is not None:
                candidates.append(image_root / path)
            candidates.append(self.data_val_root / path)

        tail = self._tail_after_data_val(path)
        if tail:
            candidates.append(self.data_val_root / tail)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        basename = path.name
        if basename:
            indexed = self.basename_index().get(basename)
            if indexed is not None and indexed.exists():
                return str(indexed.resolve())
        return None

    def basename_index(self) -> Dict[str, Path]:
        if self._basename_index is not None:
            return self._basename_index
        index: Dict[str, Path] = {}
        for path in self.data_val_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                index.setdefault(path.name, path)
        self._basename_index = index
        return index

    @staticmethod
    def _tail_after_data_val(path: Path) -> Optional[Path]:
        parts = path.parts
        if DATA_VAL_MARKER not in parts:
            return None
        marker_index = parts.index(DATA_VAL_MARKER)
        tail_parts = parts[marker_index + 1 :]
        if not tail_parts:
            return None
        return Path(*tail_parts)


class LocalGeoScoreAccuracy:
    """Dependency-light fallback mirroring the training GeoScoreAccuracy API."""

    def __init__(
        self,
        api_keys: List[str],
        max_distance: float = 2000.0,
        timeout: int = 10,
        cache_file: Optional[str] = None,
        **kwargs,
    ):
        if not api_keys:
            raise ValueError("LocalGeoScoreAccuracy requires OpenCage API keys.")
        self.api_list = api_keys
        self.api_key = api_keys[0]
        self.max_distance = float(max_distance)
        self.timeout = timeout
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.query_count = 0
        self.daily_limit = 100000
        self.current_gpu = -1
        self.quota_retry_seconds = float(os.environ.get("GEOSCORE_QUOTA_RETRY_SECONDS", "600"))
        self.rate_limit_retry_seconds = float(os.environ.get("GEOSCORE_RATE_LIMIT_RETRY_SECONDS", "30"))
        print(
            "Using local GeoScoreAccuracy fallback, "
            f"AK={self._mask_api_key(self.api_key)},max_distance={self.max_distance},"
            f"cache={self.cache_file or '<memory-only>'}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        if len(api_key) <= 8:
            return "***"
        return f"***{api_key[-4:]}"

    def _sanitize_api_error(self, error: Exception) -> str:
        message = str(error)
        for key in self.api_list:
            if key:
                message = message.replace(key, self._mask_api_key(key))
        return re.sub(r"([?&]key=)[^&\s]+", r"\1***", message)

    @staticmethod
    def _cache_key(address: str) -> str:
        normalized = address.replace(";", ",").replace("；", ",").replace("，", ",")
        normalized = re.sub(r"\s*,\s*", ",", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        return normalized

    def _read_cache_file_unlocked(self) -> Dict[str, Any]:
        if not self.cache_file or not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[GeoScoreAccuracy] failed to read cache: {self.cache_file}, error={exc}")
            return {}

    def _get_cached_geocoding(self, address: str):
        key = self._cache_key(address)
        if key in self.cache:
            return self.cache[key]
        disk_cache = self._read_cache_file_unlocked()
        if key in disk_cache:
            self.cache[key] = disk_cache[key]
            return disk_cache[key]
        return None

    def _write_cache_once(self, address: str, result: Dict[str, Any]) -> None:
        key = self._cache_key(address)
        if key in self.cache:
            return
        if not self.cache_file:
            self.cache[key] = result
            return

        import fcntl

        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        lock_file = f"{self.cache_file}.lock"
        with open(lock_file, "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            disk_cache = self._read_cache_file_unlocked()
            if key not in disk_cache:
                disk_cache[key] = result
                tmp_file = f"{self.cache_file}.tmp.{os.getpid()}"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(disk_cache, f, ensure_ascii=False, indent=2)
                os.replace(tmp_file, self.cache_file)
            self.cache = disk_cache
            fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _parse_json_object(text: str):
        if not text:
            return None
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        for candidate in re.findall(r"\{.*?\}", text, flags=re.S):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _extract_answer_blocks(completion: str) -> List[str]:
        if not completion:
            return []
        return [
            block.strip()
            for block in re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.S | re.I)
        ]

    def _extract_answer_content(self, completion: str) -> Optional[str]:
        if completion is None:
            return None

        answer_blocks = self._extract_answer_blocks(completion)
        if answer_blocks:
            answer_block = answer_blocks[-1]
            data = self._parse_json_object(answer_block)
            if isinstance(data, dict) and data.get("FinalAnswer") is not None:
                return str(data["FinalAnswer"]).strip()
            return answer_block

        data = self._parse_json_object(completion)
        if isinstance(data, dict) and data.get("FinalAnswer") is not None:
            return str(data["FinalAnswer"]).strip()

        match = re.search(r'"FinalAnswer"\s*:\s*"([^"]+)"', completion, flags=re.S)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _is_usage_limit_status(code: int) -> bool:
        return code in {402, 429, 503}

    def _sleep_for_usage_limit(self, reason: str, retry_count: int, retry_after: Optional[Any] = None) -> None:
        wait_seconds = self.rate_limit_retry_seconds
        if retry_after is not None:
            try:
                wait_seconds = max(wait_seconds, float(retry_after))
            except (TypeError, ValueError):
                pass
        if retry_count >= 5:
            wait_seconds = max(wait_seconds, self.quota_retry_seconds)
        print(f"[GeoScoreAccuracy3] OpenCage usage/rate limit: {reason}; retry after {wait_seconds:.0f}s.")
        time.sleep(wait_seconds)

    @staticmethod
    def _smart_delay() -> None:
        import random

        time.sleep(random.uniform(0, 0.2))

    def _call_opencage_geocoding(self, address: str, retry_count: int = 0) -> Dict[str, Any]:
        import requests

        address = str(address).strip() if address is not None else ""
        if not address or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", address):
            return {"status": "error", "message": "invalid geocoding query", "address": address}

        if retry_count == 0:
            cached_result = self._get_cached_geocoding(address)
            if cached_result is not None:
                return cached_result

        self.query_count += 1
        if self.query_count > self.daily_limit:
            self._sleep_for_usage_limit("daily limit estimate exceeded", retry_count)
            self.query_count = 0
            return self._call_opencage_geocoding(address, retry_count + 1)

        self._smart_delay()
        params = {
            "q": address.replace(";", ",").replace("；", ",").replace("，", ","),
            "key": self.api_key,
            "language": "en",
            "limit": 1,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            if response.status_code in {402, 429, 503}:
                self._sleep_for_usage_limit(
                    f"HTTP {response.status_code}",
                    retry_count,
                    response.headers.get("Retry-After"),
                )
                return self._call_opencage_geocoding(address, retry_count + 1)
            response.raise_for_status()
            data = response.json()
            status = data.get("status", {})
            code = status.get("code")

            if code == 200:
                results = data.get("results") or []
                if not results:
                    address_parts = address.split(";")
                    if len(address_parts) > 1:
                        shorter_address = ";".join(address_parts[:-1])
                        result = self._call_opencage_geocoding(shorter_address, retry_count)
                        self._write_cache_once(address, result)
                        return result
                    result = {"status": "error", "code": 200, "message": status.get("message"), "address": address}
                    self._write_cache_once(address, result)
                    return result

                first = results[0]
                coordinates = first["geometry"]
                api_result = {
                    "status": "success",
                    "longitude": coordinates["lng"],
                    "latitude": coordinates["lat"],
                    "confidence": first.get("confidence", 0),
                    "formatted_address": first.get("formatted", ""),
                    "components": first.get("components", {}),
                }
                self._write_cache_once(address, api_result)
                return api_result

            if self._is_usage_limit_status(int(code)):
                self._sleep_for_usage_limit(f"{code}: {status.get('message', '')}", retry_count)
                return self._call_opencage_geocoding(address, retry_count + 1)

            if code == 410:
                address_parts = address.split(";")
                if len(address_parts) > 1:
                    shorter_address = ";".join(address_parts[:-1])
                    result = self._call_opencage_geocoding(shorter_address, retry_count)
                    self._write_cache_once(address, result)
                    return result

            if code == 408 and retry_count < 5:
                time.sleep(0.1)
                return self._call_opencage_geocoding(address, retry_count + 1)

            result = {
                "status": "error",
                "code": code,
                "message": status.get("message"),
                "address": address,
            }
            self._write_cache_once(address, result)
            return result
        except requests.exceptions.RequestException as exc:
            safe_error = self._sanitize_api_error(exc)
            if retry_count < 5:
                time.sleep(0.1)
                return self._call_opencage_geocoding(address, retry_count + 1)
            result = {"status": "error", "code": 0, "message": safe_error, "address": address}
            self._write_cache_once(address, result)
            return result
        except Exception as exc:
            result = {"status": "error", "code": 999, "message": str(exc), "address": address}
            self._write_cache_once(address, result)
            return result

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return 6371.0 * c

    def _calculate_geoscore(self, distance: float) -> float:
        if distance >= self.max_distance:
            return 0.0
        return math.exp(-10.0 * (distance / self.max_distance))


class TrainingGeoScoreGeocoder:
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        ensure_repo_on_path()
        keys = resolve_api_keys(args)
        if not keys:
            raise ValueError(
                "OpenCage API keys are required. Set GEOSCORE_API_KEYS, pass --geoscore-api-keys, "
                "or provide --geoscore-api-key-config."
            )

        cache_file = args.geoscore_cache_file
        if cache_file is None:
            cache_file = output_dir / "geoscore_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        os.environ["GEOSCORE_CACHE_FILE"] = str(cache_file)

        try:
            from swift.rewards.orm import GeoScoreAccuracy
        except Exception as exc:
            print(
                "Warning: cannot import swift.rewards.orm.GeoScoreAccuracy "
                f"({type(exc).__name__}: {exc}). Falling back to local compatible implementation."
            )
            GeoScoreAccuracy = LocalGeoScoreAccuracy

        self.reward = GeoScoreAccuracy(
            api_keys=keys,
            max_distance=args.geoscore_max_distance,
            timeout=args.geocode_timeout,
            cache_file=str(cache_file),
        )
        self.cache_file = cache_file

    @staticmethod
    def _split_address_parts(final_answer: str) -> List[str]:
        if not isinstance(final_answer, str):
            return []
        parts = re.split(r"\s*[;；]\s*", final_answer.strip())
        return [part.strip() for part in parts if part and part.strip()]

    def _build_geocode_candidates(self, final_answer: str) -> List[str]:
        parts = self._split_address_parts(final_answer)
        if not parts:
            return []
        candidates: List[str] = []
        for length in range(len(parts), 0, -1):
            candidate = "; ".join(parts[:length]).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def extract_answer(self, raw_answer: Any) -> Optional[str]:
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            return None
        answer = self.reward._extract_answer_content(raw_answer)
        if isinstance(answer, str) and answer.strip():
            return normalize_answer(answer)
        return extract_final_answer(raw_answer)

    def geocode(self, final_answer: str) -> Dict[str, Any]:
        candidates = self._build_geocode_candidates(final_answer)
        if not candidates:
            return {"status": "error", "message": "invalid geocoding query", "address": final_answer}

        attempts: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            geo_result = self.reward._call_opencage_geocoding(candidate)
            lat = safe_float(geo_result.get("latitude")) if isinstance(geo_result, dict) else None
            lng = safe_float(geo_result.get("longitude")) if isinstance(geo_result, dict) else None
            if isinstance(geo_result, dict):
                attempt_record = dict(geo_result)
                attempt_record.setdefault("address", candidate)
            else:
                attempt_record = {"status": "error", "message": "invalid geocode result", "address": candidate}
            attempts.append(attempt_record)
            if lat is not None and lng is not None:
                if candidate != final_answer:
                    geo_result = dict(geo_result)
                    geo_result["resolved_from"] = candidate
                    geo_result["original_final_answer"] = final_answer
                    geo_result["fallback_attempts"] = attempts
                return geo_result
            if index < len(candidates) - 1:
                print(f"[GeoScoreGeocoder] No valid coordinates for '{candidate}', retrying with broader address.")

        last_result = attempts[-1]
        last_result = dict(last_result)
        last_result["original_final_answer"] = final_answer
        last_result["fallback_attempts"] = attempts
        return last_result

    def distance_km(self, pred_lat: float, pred_lng: float, gt_lat: float, gt_lng: float) -> float:
        return float(self.reward._calculate_distance(pred_lat, pred_lng, gt_lat, gt_lng))

    def training_reward_geoscore(self, distance_km: float) -> float:
        return float(self.reward._calculate_geoscore(distance_km))


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def try_load_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_final_answer(raw_answer: Any) -> Optional[str]:
    if not isinstance(raw_answer, str):
        return None
    text = raw_answer.strip()
    if not text:
        return None

    match = ANSWER_BLOCK_RE.search(text)
    payload = strip_code_fences(match.group(1).strip() if match else text)
    data = try_load_json(payload)
    if isinstance(data, dict):
        value = data.get("FinalAnswer") or data.get("final_answer")
        if isinstance(value, str) and value.strip():
            return normalize_answer(value)

    regex_match = FINAL_ANSWER_RE.search(payload) or FINAL_ANSWER_RE.search(text)
    if regex_match:
        try:
            decoded = json.loads(f'"{regex_match.group(1)}"')
        except json.JSONDecodeError:
            decoded = regex_match.group(1)
        return normalize_answer(decoded)
    return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def get_truth_coordinates(
    row: Dict[str, Any],
    latitude_field: str = "auto",
    longitude_field: str = "auto",
) -> Optional[Tuple[float, float]]:
    lat_keys = [latitude_field] if latitude_field != "auto" else ["latitude", "gt_latitude", "lat"]
    lon_keys = [longitude_field] if longitude_field != "auto" else ["longitude", "gt_longitude", "lon", "lng"]

    lat = None
    lon = None
    for key in lat_keys:
        lat = safe_float(row.get(key))
        if lat is not None:
            break
    for key in lon_keys:
        lon = safe_float(row.get(key))
        if lon is not None:
            break
    if lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def report_geoscore(distance_km: float) -> float:
    return 5000.0 * math.exp(-10.0 * distance_km / 18050.0)


def accuracy_at_threshold(distances: Sequence[float], threshold: float) -> float:
    if not distances:
        return 0.0
    return sum(distance <= threshold for distance in distances) / len(distances)


def compute_metrics(rows: Sequence[Dict[str, Any]], result_field: str) -> Dict[str, Any]:
    status_counts = {-1: 0, 0: 0, 1: 0}
    distances: List[float] = []
    all_sample_geoscores: List[float] = []

    for row in rows:
        result = row.get(result_field)
        if not isinstance(result, dict):
            all_sample_geoscores.append(0.0)
            continue
        status = result.get("status")
        if status in status_counts:
            status_counts[int(status)] += 1
        if status == 1:
            distance = safe_float(result.get("distance"))
            if distance is not None:
                distances.append(distance)
                all_sample_geoscores.append(report_geoscore(distance))
                continue
        all_sample_geoscores.append(0.0)

    metrics: Dict[str, Any] = {
        "total_samples": len(rows),
        "valid_predictions": len(distances),
        "status_counts": status_counts,
    }
    for metric_name, threshold in DISTANCE_THRESHOLDS.items():
        hit_count = sum(distance <= threshold for distance in distances)
        metrics[metric_name] = hit_count / len(rows) if rows else 0.0

    if not distances:
        metrics["mean_distance"] = 0.0
        metrics["median_distance"] = 0.0
    else:
        metrics["mean_distance"] = statistics.mean(distances)
        metrics["median_distance"] = statistics.median(distances)

    if all_sample_geoscores:
        metrics["mean_geoscore"] = statistics.mean(all_sample_geoscores)
        metrics["median_geoscore"] = statistics.median(all_sample_geoscores)
    else:
        metrics["mean_geoscore"] = 0.0
        metrics["median_geoscore"] = 0.0
    return metrics


def group_by_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "UNKNOWN").strip() or "UNKNOWN"
        groups.setdefault(key, []).append(row)
    return groups


def metric_lines(title: str, metrics: Dict[str, Any], elapsed: float) -> List[str]:
    return [
        "=" * 50,
        f"{title}评测结果:",
        f"总样本数: {metrics['total_samples']}",
        f"有效预测数: {metrics['valid_predictions']}",
        f"状态统计: {metrics['status_counts']}",
        f"大洲级准确率 (2500km, 失败=0): {metrics['continent_accuracy']:.4f}",
        f"国家级准确率 (750km, 失败=0): {metrics['country_accuracy']:.4f}",
        f"区域级准确率 (200km, 失败=0): {metrics['region_accuracy']:.4f}",
        f"城市级准确率 (25km, 失败=0): {metrics['city_accuracy']:.4f}",
        f"街道级准确率 (1km, 失败=0): {metrics['street_accuracy']:.4f}",
        f"平均距离 (仅有效预测): {metrics['mean_distance']:.2f} km",
        f"中位数距离 (仅有效预测): {metrics['median_distance']:.2f} km",
        f"平均GeoScore (失败=0): {metrics['mean_geoscore']:.1f}",
        f"中位数GeoScore (失败=0): {metrics['median_geoscore']:.1f}",
        f"评测耗时: {elapsed:.2f} 秒",
        "=" * 50,
    ]


def write_report(
    rows: Sequence[Dict[str, Any]],
    result_field: str,
    continent_field: str,
    report_path: Path,
) -> Dict[str, Any]:
    start = time.time()
    overall = compute_metrics(rows, result_field)
    continent_metrics = {
        continent: compute_metrics(group_rows, result_field)
        for continent, group_rows in sorted(group_by_field(rows, continent_field).items())
    }
    elapsed = time.time() - start

    lines: List[str] = []
    lines.extend(metric_lines("总体", overall, elapsed))
    domain_metrics = {
        domain: compute_metrics(group_rows, result_field)
        for domain, group_rows in sorted(group_by_field(rows, DOMAIN_FIELD).items())
    }
    for domain, metrics in domain_metrics.items():
        lines.extend(metric_lines(f"Domain {domain}", metrics, elapsed))
    for continent, metrics in continent_metrics.items():
        lines.extend(metric_lines(f"大洲 {continent}", metrics, elapsed))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    return {
        "report_path": str(report_path),
        "overall": overall,
        "domains": domain_metrics,
        "continents": continent_metrics,
        "elapsed_seconds": elapsed,
    }


def enrich_rows(
    rows: Sequence[Dict[str, Any]],
    geocoder: TrainingGeoScoreGeocoder,
    answer_field: str,
    success_field: str,
    latitude_field: str,
    longitude_field: str,
    result_field: str,
) -> Dict[str, int]:
    stats = {"geocode_success": 0, "geocode_failed": 0, "parse_failed": 0, "inference_failed": 0}
    with tqdm(rows, desc="Geocoding", unit="sample", dynamic_ncols=True) as progress:
        for row in progress:
            if success_field and not to_bool(row.get(success_field)):
                row[result_field] = {"status": -1}
                stats["inference_failed"] += 1
                progress.set_postfix(stats)
                continue

            final_answer = geocoder.extract_answer(row.get(answer_field))
            if not final_answer:
                row[result_field] = {
                    "final_answer": "",
                    "status": 0,
                    "reason": "final_answer_parse_failed",
                }
                stats["parse_failed"] += 1
                progress.set_postfix(stats)
                continue

            geo_result = geocoder.geocode(final_answer)
            if not isinstance(geo_result, dict) or geo_result.get("status") != "success":
                row[result_field] = {
                    "final_answer": final_answer,
                    "status": 0,
                    "reason": "geocode_failed",
                    "geocode_result": geo_result,
                }
                stats["geocode_failed"] += 1
                progress.set_postfix(stats)
                continue

            pred_latitude = safe_float(geo_result.get("latitude"))
            pred_longitude = safe_float(geo_result.get("longitude"))
            result: Dict[str, Any] = {
                "final_answer": final_answer,
                "status": 1,
                "pred_latitude": pred_latitude,
                "pred_longitude": pred_longitude,
                "geocode_result": geo_result,
            }

            truth = get_truth_coordinates(row, latitude_field, longitude_field)
            if pred_latitude is None or pred_longitude is None:
                result["status"] = 0
                result["reason"] = "invalid_pred_coordinates"
                stats["geocode_failed"] += 1
            elif truth is None:
                result["status"] = 0
                result["reason"] = "missing_truth_coordinates"
                stats["geocode_failed"] += 1
            else:
                distance = geocoder.distance_km(pred_latitude, pred_longitude, truth[0], truth[1])
                result["distance"] = distance
                result["geoscore"] = report_geoscore(distance)
                result["training_reward_geoscore"] = geocoder.training_reward_geoscore(distance)
                stats["geocode_success"] += 1

            row[result_field] = result
            progress.set_postfix(stats)
    return stats


def build_engine(args: argparse.Namespace):
    ensure_repo_on_path()
    model_path, model_type, template_type, torch_dtype_name = resolve_model_settings(args)
    template = build_template_if_needed(model_path, model_type, template_type, args.disable_model_thinking)
    torch_dtype = torch_dtype_from_name(torch_dtype_name)

    print(f"Loading model: {model_path}")
    print(f"infer_backend={args.infer_backend}")
    if args.adapters:
        print("Loading adapters: " + ", ".join(args.adapters))

    if args.infer_backend == "vllm":
        from swift import VllmEngine

        tensor_parallel_size = resolve_tensor_parallel_size(args)
        limit_mm_per_prompt = parse_json_dict(args.limit_mm_per_prompt, "--limit-mm-per-prompt")
        vllm_engine_kwargs = parse_json_dict(args.vllm_engine_kwargs, "--vllm-engine-kwargs") or {}
        print(
            "vLLM settings: "
            f"tensor_parallel_size={tensor_parallel_size}, "
            f"max_model_len={args.max_model_len}, "
            f"max_num_seqs={args.max_num_seqs}, "
            f"gpu_memory_utilization={args.gpu_memory_utilization}"
        )
        if vllm_engine_kwargs:
            print(f"vLLM extra engine kwargs: {json.dumps(vllm_engine_kwargs, ensure_ascii=False, sort_keys=True)}")
        return VllmEngine(
            model_path,
            template=template,
            adapters=list(args.adapters or []),
            torch_dtype=torch_dtype,
            model_type=model_type,
            template_type=template_type,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enforce_eager=args.enforce_eager,
            engine_kwargs=vllm_engine_kwargs,
        )

    from swift import TransformersEngine

    return TransformersEngine(
        model_path,
        template=template,
        adapters=list(args.adapters or []),
        max_batch_size=args.batch_size,
        torch_dtype=torch_dtype,
        model_type=model_type,
        template_type=template_type,
        device_map=args.device_map,
    )


def run_inference_for_rows(
    rows: List[Dict[str, Any]],
    engine: Any,
    args: argparse.Namespace,
    input_json: Path,
    data_val_root: Path,
    answer_field: str,
    success_field: str,
) -> InferenceResult:
    from swift import InferRequest, RequestConfig

    prompt = read_prompt(args.prompt_file)
    image_root = args.image_root.resolve() if args.image_root else input_json.parent
    resolver = ImageResolver(data_val_root)
    request_config = RequestConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    for row in rows:
        row.setdefault(answer_field, "")
        row.setdefault(success_field, False)

    interrupted = False
    indices = list(range(len(rows)))
    with tqdm(total=len(rows), desc=f"Inferring {input_json.stem}", unit="item", dynamic_ncols=True) as pbar:
        for batch_indices in chunked(indices, max(1, args.batch_size)):
            requests = []
            valid_indices: List[int] = []
            for index in batch_indices:
                row = rows[index]
                image_path = resolver.resolve(row.get(args.image_field), image_root)
                if image_path is None:
                    row[answer_field] = ""
                    row[success_field] = False
                    row[f"{success_field}_error"] = f"missing image path from field {args.image_field}"
                    continue
                requests.append(
                    InferRequest(
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": args.user_prompt},
                        ],
                        images=[image_path],
                    )
                )
                valid_indices.append(index)

            if not requests:
                pbar.update(len(batch_indices))
                continue

            try:
                with batch_timeout(args.batch_timeout_sec):
                    responses = infer_batch(engine, requests, request_config)
            except BatchTimeoutError as exc:
                print(f"Batch timeout at index {batch_indices[0]}: {exc}", file=sys.stderr)
                interrupted = True
                break
            except Exception as exc:
                print(f"Batch failed at index {batch_indices[0]}: {type(exc).__name__}: {exc}", file=sys.stderr)
                interrupted = True
                break

            for response, index in zip(responses, valid_indices):
                if isinstance(response, Exception):
                    rows[index][answer_field] = ""
                    rows[index][success_field] = False
                    rows[index][f"{success_field}_error"] = str(response)
                    continue
                try:
                    content = response.choices[0].message.content
                except Exception as exc:
                    content = ""
                    rows[index][f"{success_field}_error"] = str(exc)
                rows[index][answer_field] = content or ""
                rows[index][success_field] = bool(content)
            pbar.update(len(batch_indices))

    return InferenceResult(rows=rows, answer_field=answer_field, success_field=success_field, completed=not interrupted)


def run_inference(args: argparse.Namespace, output_dir: Path) -> List[Path]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    input_jsons = resolve_input_jsons(args)
    print_input_summary(input_jsons, args.limit)
    data_val_root = get_data_val_root(args.data_root)
    metadata_index = MetadataIndex(data_val_root)
    answer_field, success_field = field_names(args)

    engine = build_engine(args)
    output_paths: List[Path] = []
    try:
        for input_json in input_jsons:
            rows = load_json_rows(input_json, args.limit)
            annotate_domain(rows, input_json, data_val_root)
            if not args.no_metadata_patch:
                metadata_index.patch_rows(rows)
            result = run_inference_for_rows(
                rows=rows,
                engine=engine,
                args=args,
                input_json=input_json,
                data_val_root=data_val_root,
                answer_field=answer_field,
                success_field=success_field,
            )
            output_json = output_dir / f"{safe_slug(input_json.stem)}_{safe_slug(answer_field)}_predictions.json"
            write_json(output_json, result.rows)
            print(f"Inference output written to: {output_json}")
            print(f"answer_field={result.answer_field} success_field={result.success_field} completed={result.completed}")
            output_paths.append(output_json)
    finally:
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return output_paths


def score_files(
    args: argparse.Namespace,
    output_dir: Path,
    input_jsons: Sequence[Path],
    answer_field: str,
    success_field: str,
) -> Tuple[List[Path], str]:
    if getattr(args, "command", "") == "score":
        print_input_summary(input_jsons, args.limit)
    data_val_root = get_data_val_root(args.data_root)
    metadata_index = MetadataIndex(data_val_root)
    geocoder = TrainingGeoScoreGeocoder(args, output_dir)
    result_field = f"{answer_field}{args.result_suffix}"
    enriched_paths: List[Path] = []
    combined_rows: List[Dict[str, Any]] = []

    for input_json in input_jsons:
        rows = load_json_rows(input_json, args.limit)
        annotate_domain(rows, input_json, data_val_root)
        if not args.no_metadata_patch:
            metadata_index.patch_rows(rows)
        stats = enrich_rows(
            rows=rows,
            geocoder=geocoder,
            answer_field=answer_field,
            success_field=success_field,
            latitude_field=args.latitude_field,
            longitude_field=args.longitude_field,
            result_field=result_field,
        )
        enriched_path = output_dir / f"{safe_slug(input_json.stem)}_{safe_slug(result_field)}.json"
        report_path = output_dir / f"{safe_slug(input_json.stem)}_{safe_slug(result_field)}_report.txt"
        write_json(enriched_path, rows)
        write_report(rows, result_field, args.continent_field, report_path)
        print(f"Enriched JSON written to: {enriched_path}")
        print(f"Report written to: {report_path}")
        print(f"Geocode stats: {stats}")
        enriched_paths.append(enriched_path)
        combined_rows.extend(rows)

    if len(input_jsons) > 1:
        combined_report = output_dir / f"combined_{safe_slug(result_field)}_report.txt"
        write_report(combined_rows, result_field, args.continent_field, combined_report)
        print(f"Combined report written to: {combined_report}")
    return enriched_paths, result_field


def report_files(args: argparse.Namespace, output_dir: Path) -> None:
    input_jsons = resolve_input_jsons(args)
    print_input_summary(input_jsons, args.limit)
    data_val_root = get_data_val_root(args.data_root)
    metadata_index = MetadataIndex(data_val_root)
    combined_rows: List[Dict[str, Any]] = []
    for input_json in input_jsons:
        rows = load_json_rows(input_json, args.limit)
        annotate_domain(rows, input_json, data_val_root)
        if not args.no_metadata_patch:
            metadata_index.patch_rows(rows)
        report_path = output_dir / f"{safe_slug(input_json.stem)}_{safe_slug(args.result_field)}_report.txt"
        write_report(rows, args.result_field, args.continent_field, report_path)
        combined_rows.extend(rows)
    if len(input_jsons) > 1:
        combined_report = output_dir / f"combined_{safe_slug(args.result_field)}_report.txt"
        write_report(combined_rows, args.result_field, args.continent_field, combined_report)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    args = parse_args()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    setup_logging(output_dir)

    print(f"command={args.command}")
    print(f"data_root={args.data_root.resolve()}")
    print(f"output_dir={output_dir}")

    if args.command == "infer":
        run_inference(args, output_dir)
        return

    if args.command == "score":
        input_jsons = resolve_input_jsons(args)
        score_files(args, output_dir, input_jsons, args.answer_field, args.success_field)
        return

    if args.command == "report":
        report_files(args, output_dir)
        return

    prediction_paths = run_inference(args, output_dir)
    answer_field, success_field = field_names(args)
    score_files(args, output_dir, prediction_paths, answer_field, success_field)


if __name__ == "__main__":
    main()
