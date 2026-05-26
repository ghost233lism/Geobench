#!/usr/bin/env python3
"""Sample geolocation VLM outputs, score them, and downsample images by value."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
DATA_CLEAN_DIR = Path(__file__).resolve().parent
DEFAULT_TEXT_GEOCODE_PROMPT_FILE = DATA_CLEAN_DIR / "prompt_text_geocode.txt"
DEFAULT_LATLON_GEOSCORE_PROMPT_FILE = DATA_CLEAN_DIR / "prompt_latlon_geoscore.txt"
DEFAULT_JUDGE_BANDED_PROMPT_FILE = DATA_CLEAN_DIR / "prompt_judge_banded.txt"
DEFAULT_JUDGE_EXPONENTIAL_PROMPT_FILE = DATA_CLEAN_DIR / "prompt_judge_exponential.txt"
DEFAULT_MODEL_ROOT = Path("/nfs/sunboyuan/model")


@dataclass
class DatasetItem:
    item_id: str
    image_path: str
    relative_path: str
    gt_text: Optional[str] = None
    gt_latitude: Optional[float] = None
    gt_longitude: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SampleResult:
    item_id: str
    image_path: str
    relative_path: str
    model_name: str
    sample_index: int
    reward_mode: str
    raw_output: str
    parsed_answer: Optional[str]
    parsed_latitude: Optional[float]
    parsed_longitude: Optional[float]
    reward: float
    reward_details: Dict[str, Any]
    gt_text: Optional[str]
    gt_latitude: Optional[float]
    gt_longitude: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM sampling, compute geolocation rewards, and downsample images by value."
    )
    parser.add_argument("--data-dir", required=True, help="Root folder containing images.")
    parser.add_argument("--output-dir", required=True, help="Directory for intermediate metadata and outputs.")
    parser.add_argument(
        "--annotation-file",
        help="Optional json/jsonl/csv manifest. If omitted, the script tries same-basename .json sidecars.",
    )
    parser.add_argument(
        "--image-field",
        default="auto",
        help="Image path field in annotation file. 'auto' tries common names.",
    )
    parser.add_argument(
        "--gt-text-field",
        default="auto",
        help="GT text field in annotation file. 'auto' tries common names.",
    )
    parser.add_argument(
        "--gt-lat-field",
        default="auto",
        help="GT latitude field in annotation file. 'auto' tries common names.",
    )
    parser.add_argument(
        "--gt-lon-field",
        default="auto",
        help="GT longitude field in annotation file. 'auto' tries common names.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model directories or names under /nfs/sunboyuan/model.",
    )
    parser.add_argument(
        "--samples-per-model",
        help="Comma-separated sample count per model. Overrides --total-samples.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=8,
        help="Total samples per image across all generation models when --samples-per-model is omitted.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="vLLM data parallel size. For example, 4 GPUs can be split into 2 groups with tp=2, dp=2.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["text_geocode", "latlon_geoscore", "llm_judge"],
        required=True,
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for reward-mode=llm_judge. Loaded only after generation is complete.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--judge-prompt-mode",
        choices=["banded", "exponential"],
        default="banded",
        help="Which judge prompt template to use for reward-mode=llm_judge.",
    )
    parser.add_argument(
        "--disable-model-thinking",
        action="store_true",
        help="Disable native thinking mode for models/templates that support it, such as InternVL3.5 and Qwen3/Qwen3.5.",
    )
    parser.add_argument(
        "--opencage-api-keys",
        default="",
        help="Comma-separated OpenCage API keys. Needed for geocoding modes when text must be geocoded.",
    )
    parser.add_argument("--max-distance", type=float, default=18050.0)
    parser.add_argument("--confidence-threshold", type=int, default=0)
    parser.add_argument(
        "--prompt-file",
        help="Optional override for the generation prompt file. Defaults to a standalone prompt chosen by reward mode.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, help="Process at most this many images after discovery.")
    parser.add_argument("--keep-count", type=int, help="Downsample to this many images. Default keeps all.")
    parser.add_argument(
        "--mid-target",
        type=float,
        help="m_mu in caculate_p.md. Default uses the global reward mean.",
    )
    parser.add_argument("--tau-mu", type=float, default=1.0)
    parser.add_argument("--tau-sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--variance-mode",
        choices=["exp_std", "percentile"],
        default="exp_std",
        help="Which discriminability score to use from caculate_p.md.",
    )
    parser.add_argument(
        "--export-mode",
        choices=["symlink", "copy", "manifest_only"],
        default="symlink",
        help="How to materialize selected images while preserving folder structure.",
    )
    parser.add_argument(
        "--selected-root-name",
        default="selected_images",
        help="Subdirectory name under output-dir for exported selected images.",
    )
    parser.add_argument(
        "--geocode-cache-file",
        help="Optional cache file path. Defaults to output-dir/geocode_cache.json.",
    )
    parser.add_argument(
        "--retry-error-cache",
        action="store_true",
        help="Retry cached geocode entries whose cached payload status is error.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Pass disable_custom_all_reduce=True to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM.",
    )
    return parser.parse_args()


def ensure_repo_on_path() -> None:
    repo_root = Path("/nfs/sunboyuan/Geobench/ms-swift")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def set_runtime_env(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def get_visible_device_count(cuda_visible_devices: str) -> int:
    return len([device for device in parse_csv_list(cuda_visible_devices) if device])


def validate_parallel_config(args: argparse.Namespace) -> None:
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1")
    if args.data_parallel_size < 1:
        raise ValueError("--data-parallel-size must be >= 1")
    visible_device_count = get_visible_device_count(args.cuda_visible_devices)
    required_device_count = args.tensor_parallel_size * args.data_parallel_size
    if visible_device_count < required_device_count:
        raise ValueError(
            "Not enough visible GPUs for the requested parallelism: "
            f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices!r} exposes {visible_device_count} device(s), "
            f"but tensor_parallel_size ({args.tensor_parallel_size}) * "
            f"data_parallel_size ({args.data_parallel_size}) = {required_device_count}."
        )


def build_vllm_engine_kwargs(
    args: argparse.Namespace,
    model_profile: Dict[str, Any],
) -> Dict[str, Any]:
    engine_kwargs: Dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
    }
    if model_profile.get("limit_mm_per_prompt"):
        engine_kwargs["limit_mm_per_prompt"] = model_profile["limit_mm_per_prompt"]
    if args.disable_custom_all_reduce:
        engine_kwargs["disable_custom_all_reduce"] = True
    if args.data_parallel_size > 1:
        engine_kwargs["use_async_engine"] = True
        engine_kwargs["engine_kwargs"] = {"data_parallel_size": args.data_parallel_size}
    return engine_kwargs


def maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip("/")) or "model"


def stable_item_id(relative_path: str) -> str:
    digest = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
    return f"{sanitize_name(Path(relative_path).stem)}_{digest[:12]}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def default_generation_prompt_file(reward_mode: str) -> Path:
    if reward_mode == "latlon_geoscore":
        return DEFAULT_LATLON_GEOSCORE_PROMPT_FILE
    return DEFAULT_TEXT_GEOCODE_PROMPT_FILE


def default_judge_prompt_file(judge_prompt_mode: str) -> Path:
    if judge_prompt_mode == "exponential":
        return DEFAULT_JUDGE_EXPONENTIAL_PROMPT_FILE
    return DEFAULT_JUDGE_BANDED_PROMPT_FILE


def should_strip_think_tokens(model_profile: Dict[str, Any]) -> bool:
    swift_model_type = str(model_profile.get("swift_model_type") or "").lower()
    swift_template_type = str(model_profile.get("swift_template_type") or "").lower()
    model_dir_name = str(model_profile.get("model_dir_name") or "").lower()
    model_path = str(model_profile.get("model_path") or "").lower()

    intern35_markers = (
        "internvl3_5",
        "internvl3.5",
        "intern3_5vl",
        "intern3.5vl",
    )
    combined = " ".join((swift_model_type, swift_template_type, model_dir_name, model_path))
    return any(marker in combined for marker in intern35_markers)


def adapt_prompt_for_model(system_prompt: str, model_profile: Dict[str, Any]) -> str:
    if should_strip_think_tokens(model_profile):
        return system_prompt.replace("<think>", "").replace("</think>", "")
    return system_prompt


def build_infer_template(model_profile: Dict[str, Any], disable_model_thinking: bool):
    if not disable_model_thinking:
        return None
    ensure_repo_on_path()
    from swift.model import get_processor
    from swift.template import get_template

    processor = get_processor(
        model_id_or_path=model_profile["model_path"],
        model_type=model_profile.get("swift_model_type"),
        download_model=True,
    )
    return get_template(
        processor,
        template_type=model_profile.get("swift_template_type"),
        enable_thinking=False,
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def find_model_root(path: Path) -> Optional[Path]:
    if (path / "config.json").exists():
        return path
    direct_children = sorted(
        [child for child in path.iterdir() if child.is_dir() and (child / "config.json").exists()],
        key=lambda child: child.name,
    )
    if direct_children:
        return direct_children[0]
    recursive = sorted(path.rglob("config.json"))
    if recursive:
        return recursive[0].parent
    return None


def resolve_model_path(model_name: str) -> str:
    path = Path(model_name)
    if path.exists():
        model_root = find_model_root(path)
        if model_root is None:
            raise FileNotFoundError(f"Cannot find config.json under model path: {model_name}")
        return str(model_root.resolve())
    candidate = DEFAULT_MODEL_ROOT / model_name
    if candidate.exists():
        model_root = find_model_root(candidate)
        if model_root is None:
            raise FileNotFoundError(f"Cannot find config.json under model path: {candidate}")
        return str(model_root.resolve())
    raise FileNotFoundError(f"Cannot resolve model path: {model_name}")


def detect_model_profile(model_path: str) -> Dict[str, Any]:
    root = Path(model_path)
    config = load_json(root / "config.json", {}) or {}
    processor_cfg = load_json(root / "processor_config.json", {}) or {}
    preprocessor_cfg = load_json(root / "preprocessor_config.json", {}) or {}
    tokenizer_cfg = load_json(root / "tokenizer_config.json", {}) or {}
    model_name = root.name.lower()

    architectures = [str(x).lower() for x in config.get("architectures", [])]
    model_type = str(config.get("model_type", "")).lower()
    processor_class = str(processor_cfg.get("processor_class") or preprocessor_cfg.get("processor_class") or "").lower()
    image_token_like = any(
        key in config
        for key in ("image_token_id", "image_token_index", "boi_token_index", "vision_start_token_id")
    )
    is_multimodal = bool(
        image_token_like
        or "vision_config" in config
        or "image_processor_type" in preprocessor_cfg
        or "processor_class" in processor_cfg
        or "vision_start_token_id" in tokenizer_cfg.get("added_tokens_decoder", {})
    )

    family = "generic"
    if "internvl" in model_type or "internvl" in processor_class or any("internvl" in x for x in architectures):
        family = "internvl"
    elif "qwen3_vl" in model_type or any("qwen3vl" in x for x in architectures):
        family = "qwen3_vl"
    elif model_type == "qwen3_5":
        family = "qwen3_5"
    elif "gemma3" in model_type or any("gemma3" in x for x in architectures):
        family = "gemma3"
    elif "qwen3" in model_type or any("qwen3" in x for x in architectures):
        family = "qwen3"

    swift_model_type: Optional[str] = None
    swift_template_type: Optional[str] = None
    config_template = str(config.get("template", "")).strip()

    if family == "internvl":
        if "3_5_gpt" in model_name or config_template == "internvl3_5_gpt":
            swift_model_type = "internvl3_5_gpt"
            swift_template_type = "internvl3_5_gpt"
        elif "3_5" in model_name or "3.5" in model_name:
            swift_model_type = "internvl3_5"
            swift_template_type = "internvl3_5"
        elif "internvl3" in model_name:
            swift_model_type = "internvl3"
            swift_template_type = "internvl2_5"
        elif "2_5" in model_name or "2.5" in model_name or config_template == "internvl2_5":
            swift_model_type = "internvl2_5"
            swift_template_type = "internvl2_5"
        elif config_template in {"internvl", "internvl2", "internvl2_5", "internvl3_5", "internvl3_5_gpt", "internvl_hf"}:
            swift_template_type = config_template
            if config_template == "internvl3_5":
                swift_model_type = "internvl3_5"
            elif config_template == "internvl3_5_gpt":
                swift_model_type = "internvl3_5_gpt"
            elif config_template == "internvl2_5":
                swift_model_type = "internvl2_5"
            elif config_template == "internvl2":
                swift_model_type = "internvl2"
            elif config_template == "internvl_hf":
                swift_model_type = "internvl_hf"
            else:
                swift_model_type = "internvl"
        else:
            swift_model_type = "internvl3_5"
            swift_template_type = "internvl3_5"
    elif family == "qwen3_vl":
        swift_model_type = "qwen3_vl"
        swift_template_type = "qwen3_vl"
    elif family == "qwen3_5":
        swift_model_type = "qwen3_5"
        swift_template_type = "qwen3_5"
    elif family == "gemma3":
        if is_multimodal:
            swift_model_type = "gemma3_vision"
            swift_template_type = "gemma3_vision"
        else:
            swift_model_type = "gemma3_text"
            swift_template_type = "gemma3_text"
    elif family == "qwen3":
        swift_model_type = "qwen3"
        swift_template_type = "qwen3"

    profile = {
        "model_path": model_path,
        "model_dir_name": root.name,
        "family": family,
        "hf_model_type": model_type,
        "architectures": config.get("architectures", []),
        "processor_class": processor_class,
        "is_multimodal": is_multimodal,
        "supports_images": is_multimodal,
        "limit_mm_per_prompt": {"image": 1, "video": 0} if is_multimodal else None,
        "swift_model_type": swift_model_type,
        "swift_template_type": swift_template_type,
        "config_template": config_template or None,
    }
    return profile


def discover_images(data_dir: Path) -> List[Path]:
    return sorted([path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])


def common_field(row: Dict[str, Any], override: str, candidates: Sequence[str]) -> Optional[str]:
    if override != "auto":
        return override if override in row else None
    for key in candidates:
        if key in row:
            return key
    return None


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "annotations", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON manifest format: {path}")


def load_annotation_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_json_or_jsonl(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported annotation file: {path}")


def normalize_image_ref(data_dir: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())
    return str((data_dir / path).resolve())


def load_annotations(args: argparse.Namespace, data_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not args.annotation_file:
        return {}
    rows = load_annotation_rows(Path(args.annotation_file))

    annotations: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_key = common_field(
            row,
            args.image_field,
            ("image", "images", "image_path", "path", "file_name", "filename", "relative_path"),
        )
        if not image_key:
            raise ValueError(f"Missing image field in annotation row: {row}")
        image_value = row[image_key]
        if isinstance(image_value, list):
            image_value = image_value[0]
        image_path = normalize_image_ref(data_dir, str(image_value))
        text_key = common_field(
            row,
            args.gt_text_field,
            ("gt_text", "ground_truth", "ground_truth_text", "label", "answer", "solution", "location", "gt"),
        )
        lat_key = common_field(row, args.gt_lat_field, ("gt_latitude", "latitude", "lat", "gt_lat"))
        lon_key = common_field(row, args.gt_lon_field, ("gt_longitude", "longitude", "lon", "lng", "gt_lon", "gt_lng"))
        
        annotations[image_path] = {
            "gt_text": row.get(text_key) if text_key else None,
            "gt_latitude": maybe_float(row.get(lat_key)) if lat_key else None,
            "gt_longitude": maybe_float(row.get(lon_key)) if lon_key else None,
            "metadata": row,
        }
    return annotations


def load_sidecar_annotation(image_path: Path) -> Dict[str, Any]:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    row = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        return {}
    text_key = common_field(
        row,
        "auto",
        ("gt_text", "ground_truth", "ground_truth_text", "label", "answer", "solution", "location", "gt"),
    )
    lat_key = common_field(row, "auto", ("gt_latitude", "latitude", "lat", "gt_lat"))
    lon_key = common_field(row, "auto", ("gt_longitude", "longitude", "lon", "lng", "gt_lon", "gt_lng"))
    return {
        "gt_text": row.get(text_key) if text_key else None,
        "gt_latitude": maybe_float(row.get(lat_key)) if lat_key else None,
        "gt_longitude": maybe_float(row.get(lon_key)) if lon_key else None,
        "metadata": row,
    }


def build_dataset(args: argparse.Namespace) -> List[DatasetItem]:
    data_dir = Path(args.data_dir).resolve()
    images = discover_images(data_dir)
    annotations = load_annotations(args, data_dir)
    # print(f"annotations: {annotations}")
    items: List[DatasetItem] = []
    for image_path in tqdm(images, desc="Scanning images", unit="image"):
        resolved = str(image_path.resolve())
        # print(f"resolved: {resolved}")
        ann = annotations.get(resolved) or load_sidecar_annotation(image_path)
        # print(f"ann: {ann}")
        # exit()
        gt_text = ann.get("gt_text")
        # print(f"gt_text: {gt_text}")
        gt_lat = ann.get("gt_latitude")
        # print(f"gt_lat: {gt_lat}")
        gt_lon = ann.get("gt_longitude")
        # print(f"gt_lon: {gt_lon}")
        # exit()
        if gt_text is None and (gt_lat is None or gt_lon is None):
            continue
        rel_path = str(image_path.relative_to(data_dir))
        items.append(
            DatasetItem(
                item_id=stable_item_id(rel_path),
                image_path=resolved,
                relative_path=rel_path,
                gt_text=gt_text,
                gt_latitude=gt_lat,
                gt_longitude=gt_lon,
                metadata=ann.get("metadata"),
            )
        )
    if args.max_images is not None:
        items = items[: args.max_images]
    if not items:
        raise ValueError("No usable images found. Provide annotations or same-basename sidecar JSON files.")
    return items


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def extract_answer_block(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_string_field(raw_text: str, field_name: str) -> Optional[str]:
    if not raw_text:
        return None
    patterns = [
        rf"[\"']?{re.escape(field_name)}[\"']?\s*[:=]\s*\"([^\"\n<>]+)\"",
        rf"['\"]?{re.escape(field_name)}['\"]?\s*[:=]\s*'([^'\n<>]+)'",
        rf"[\"']?{re.escape(field_name)}[\"']?\s*[:=]\s*([^,\n}}<>]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            value = strip_wrapping_quotes(match.group(1)).strip(" ,")
            if value:
                return value
    return None


def extract_numeric_field(raw_text: str, field_names: Sequence[str]) -> Optional[float]:
    if not raw_text:
        return None
    for field_name in field_names:
        match = re.search(
            rf"[\"']?{re.escape(field_name)}[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if match:
            value = maybe_float(match.group(1))
            if value is not None:
                return value
    return None


def extract_answer_text(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None
    answer_block = extract_answer_block(raw_output)
    if not answer_block:
        answer_block = raw_output.strip()
    json_text = extract_json_block(answer_block)
    if isinstance(json_text, dict):
        final_answer = json_text.get("FinalAnswer")
        if isinstance(final_answer, str):
            final_answer = final_answer.strip()
            if final_answer:
                return final_answer
    final_answer = extract_string_field(answer_block, "FinalAnswer")
    if final_answer:
        return final_answer
    # Fallback for outputs like: <answer>Country; Region; Specific Location</answer>
    compact = strip_wrapping_quotes(answer_block).strip()
    if compact and "<" not in compact and ">" not in compact and len(compact) <= 300:
        return compact
    return None


def extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.extend(re.findall(r"(\{.*?\})", raw_text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def is_placeholder_judge_json(payload: Dict[str, Any]) -> bool:
    verdict = str(payload.get("verdict", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    score = payload.get("score")
    if verdict == "..." or reason == "...":
        return True
    if isinstance(score, str) and score.strip() in {"...", "0.0"} and (verdict == "" or reason == ""):
        return True
    return False


def extract_last_valid_judge_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.extend(re.findall(r"(\{.*?\})", raw_text, flags=re.DOTALL))

    valid_payloads: List[Dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "score" not in parsed or "verdict" not in parsed or "reason" not in parsed:
            continue
        score = maybe_float(parsed.get("score"))
        if score is None:
            continue
        parsed["score"] = score
        if is_placeholder_judge_json(parsed):
            continue
        valid_payloads.append(parsed)
    if valid_payloads:
        return valid_payloads[-1]

    score = extract_numeric_field(raw_text, ("score",))
    verdict = extract_string_field(raw_text, "verdict")
    reason = extract_string_field(raw_text, "reason")
    if score is None or not verdict or not reason:
        return None
    payload = {"score": score, "verdict": verdict, "reason": reason}
    if is_placeholder_judge_json(payload):
        return None
    return payload


def extract_latlon(raw_output: str) -> Tuple[Optional[float], Optional[float]]:
    answer_block = extract_answer_block(raw_output)
    json_block = extract_json_block(raw_output)
    for source in (extract_json_block(answer_block or ""), json_block):
        if isinstance(source, dict):
            lat = maybe_float(source.get("latitude", source.get("lat")))
            lon = maybe_float(source.get("longitude", source.get("lon", source.get("lng"))))
            if lat is not None and lon is not None:
                return lat, lon
    text = answer_block or raw_output
    lat = extract_numeric_field(text, ("latitude", "lat"))
    lon = extract_numeric_field(text, ("longitude", "lon", "lng"))
    if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180:
        return lat, lon
    pair_patterns = [
        r"latitude\s*[:=]\s*(-?\d+(?:\.\d+)?)\D+longitude\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"lat\s*[:=]\s*(-?\d+(?:\.\d+)?)\D+lon(?:gitude)?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
    ]
    for pattern in pair_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return maybe_float(match.group(1)), maybe_float(match.group(2))
    numeric = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numeric) >= 2:
        lat, lon = maybe_float(numeric[0]), maybe_float(numeric[1])
        if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon
    return None, None


def build_judge_prompt(prompt_template: str, prediction: str, ground_truth: str, max_distance: float) -> str:
    return (
        prompt_template.replace("__MAX_DISTANCE_KM__", f"{max_distance:g}")
        .replace("__PREDICTION__", prediction)
        .replace("__GROUND_TRUTH__", ground_truth)
    )


def geodesic_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return 6371.0 * c


def calculate_geoscore(distance_km: float, max_distance: float) -> float:
    if distance_km >= max_distance:
        return 0.0
    return float(math.exp(-10 * (distance_km / max_distance)))


def build_geo_point(
    latitude: float,
    longitude: float,
    source: str,
    raw_geo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    point: Dict[str, Any] = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "source": source,
    }
    if raw_geo:
        for key in ("status", "confidence", "formatted", "formatted_address"):
            value = raw_geo.get(key)
            if value is not None:
                point[key] = value
    return point


def build_geoscore_details(
    pred_geo: Dict[str, Any],
    gt_geo: Dict[str, Any],
    distance_km: float,
    geoscore: float,
) -> Dict[str, Any]:
    return {
        "pred_geo": pred_geo,
        "gt_geo": gt_geo,
        "distance_km": float(distance_km),
        "geoscore": float(geoscore),
    }


class OpenCageCache:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        if cache_file.exists():
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    def get(self, address: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(address.strip())

    def set(self, address: str, payload: Dict[str, Any]) -> None:
        self.cache[address.strip()] = payload

    def flush(self) -> None:
        write_json(self.cache_file, self.cache)


class OpenCageWrapper:
    def __init__(self, api_keys: List[str], cache: OpenCageCache, retry_error_cache: bool = False):
        if not api_keys:
            raise ValueError("OpenCage API keys are required for geocoding.")
        self.api_keys = api_keys
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"
        self.cache = cache
        self._cursor = 0
        self.retry_error_cache = retry_error_cache
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies.update(self._resolve_proxies())

    @staticmethod
    def _resolve_proxies() -> Dict[str, str]:
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
        proxies: Dict[str, str] = {}
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        if proxies:
            return proxies
        if all_proxy and not all_proxy.lower().startswith("socks"):
            return {"http": all_proxy, "https": all_proxy}
        return {}

    def geocode(self, address: str) -> Optional[Dict[str, Any]]:
        address = address.strip()
        if not address:
            return None
        cached = self.cache.get(address)
        if cached is not None:
            if cached.get("status") != "error" or not self.retry_error_cache:
                return cached
        api_key = self.api_keys[self._cursor % len(self.api_keys)]
        self._cursor += 1
        try:
            response = self.session.get(
                self.base_url,
                params={
                    "q": address.replace(";", ",").replace("；", ",").replace("，", ","),
                    "key": api_key,
                    "language": "en",
                    "limit": 1,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status", {}).get("code") == 200 and data.get("results"):
                result = data["results"][0]
                geometry = result["geometry"]
                payload = {
                    "status": "success",
                    "latitude": geometry["lat"],
                    "longitude": geometry["lng"],
                    "confidence": result.get("confidence", 0),
                    "formatted_address": result.get("formatted"),
                }
            else:
                payload = {
                    "status": "error",
                    "code": data.get("status", {}).get("code"),
                    "message": data.get("status", {}).get("message"),
                }
        except Exception as exc:
            payload = {"status": "error", "message": str(exc)}
        self.cache.set(address, payload)
        return payload


def make_requests(
    items: Sequence[DatasetItem],
    system_prompt: str,
    model_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    requests_payload: List[Dict[str, Any]] = []
    if not model_profile.get("supports_images", False):
        raise ValueError(
            f"Generation model does not look like a multimodal image model: {model_profile.get('model_path')}"
        )
    for item in items:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Locate this image."}]
        requests_payload.append({"messages": messages, "images": [item.image_path]})
    return requests_payload


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def run_generation_for_model(
    args: argparse.Namespace,
    model_profile: Dict[str, Any],
    model_name: str,
    items: Sequence[DatasetItem],
    system_prompt: str,
    samples_per_image: int,
) -> List[SampleResult]:
    ensure_repo_on_path()
    from swift.infer_engine import RequestConfig, VllmEngine

    template = build_infer_template(model_profile, args.disable_model_thinking)
    engine_kwargs = build_vllm_engine_kwargs(args, model_profile)
    engine = VllmEngine(
        model_profile["model_path"],
        template=template,
        model_type=model_profile.get("swift_model_type"),
        template_type=model_profile.get("swift_template_type"),
        **engine_kwargs,
    )
    request_config = RequestConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    results: List[SampleResult] = []
    batched_items = list(chunked(list(items), args.infer_batch_size))
    total_images = len(items)
    with tqdm(total=total_images, desc=f"Generating [{model_name}]", unit="image") as pbar:
        for batch in batched_items:
            payloads = []
            expanded_items: List[Tuple[DatasetItem, int]] = []
            base_payloads = make_requests(batch, system_prompt, model_profile)
            for item, base_payload in zip(batch, base_payloads):
                for sample_index in range(samples_per_image):
                    payloads.append(dict(base_payload))
                    expanded_items.append((item, sample_index))
            responses = engine.infer(payloads, request_config=request_config)
            for response, (item, sample_index) in zip(responses, expanded_items):
                raw_output = response.choices[0].message.content if not isinstance(response, Exception) else ""
                parsed_answer = extract_answer_text(raw_output)
                pred_lat, pred_lon = extract_latlon(raw_output) if args.reward_mode == "latlon_geoscore" else (None, None)
                results.append(
                    SampleResult(
                        item_id=item.item_id,
                        image_path=item.image_path,
                        relative_path=item.relative_path,
                        model_name=model_name,
                        sample_index=sample_index,
                        reward_mode=args.reward_mode,
                        raw_output=raw_output,
                        parsed_answer=parsed_answer,
                        parsed_latitude=pred_lat,
                        parsed_longitude=pred_lon,
                        reward=0.0,
                        reward_details={},
                        gt_text=item.gt_text,
                        gt_latitude=item.gt_latitude,
                        gt_longitude=item.gt_longitude,
                    )
                )
            for _ in batch:
                pbar.update(1)
    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return results


def score_text_geocode(
    samples: List[SampleResult],
    geocoder: OpenCageWrapper,
    max_distance: float,
    confidence_threshold: int,
) -> None:
    for sample in tqdm(samples, desc="Scoring text_geocode", unit="sample"):
        pred_text = sample.parsed_answer
        if not pred_text:
            sample.reward = 0.0
            sample.reward_details = {"error": "missing_answer"}
            continue
        pred_geo = geocoder.geocode(pred_text)
        if not pred_geo or pred_geo.get("status") != "success":
            sample.reward = 0.0
            sample.reward_details = {"error": "pred_geocode_failed", "pred_geo": pred_geo}
            continue
        if pred_geo.get("confidence", 0) < confidence_threshold:
            sample.reward = 0.0
            sample.reward_details = {"error": "pred_low_confidence", "pred_geo": pred_geo}
            continue
        gt_geo = None
        if sample.gt_latitude is not None and sample.gt_longitude is not None:
            gt_geo = {"status": "success", "latitude": sample.gt_latitude, "longitude": sample.gt_longitude}
        elif sample.gt_text:
            gt_geo = geocoder.geocode(sample.gt_text)
        if not gt_geo or gt_geo.get("status") != "success":
            sample.reward = 0.0
            sample.reward_details = {"error": "gt_geocode_failed", "pred_geo": pred_geo, "gt_geo": gt_geo}
            continue
        if gt_geo.get("confidence", confidence_threshold) < confidence_threshold:
            sample.reward = 0.0
            sample.reward_details = {"error": "gt_low_confidence", "pred_geo": pred_geo, "gt_geo": gt_geo}
            continue
        distance = geodesic_km(
            pred_geo["latitude"], pred_geo["longitude"], gt_geo["latitude"], gt_geo["longitude"]
        )
        sample.reward = calculate_geoscore(distance, max_distance)
        pred_point = build_geo_point(
            pred_geo["latitude"],
            pred_geo["longitude"],
            source="geocoded_prediction_text",
            raw_geo=pred_geo,
        )
        gt_point = build_geo_point(
            gt_geo["latitude"],
            gt_geo["longitude"],
            source="annotation_latlon" if sample.gt_latitude is not None and sample.gt_longitude is not None else "geocoded_ground_truth_text",
            raw_geo=gt_geo,
        )
        sample.reward_details = build_geoscore_details(pred_point, gt_point, distance, sample.reward)


def score_latlon_geoscore(
    samples: List[SampleResult],
    geocoder: Optional[OpenCageWrapper],
    max_distance: float,
) -> None:
    for sample in tqdm(samples, desc="Scoring latlon_geoscore", unit="sample"):
        pred_lat = sample.parsed_latitude
        pred_lon = sample.parsed_longitude
        if pred_lat is None or pred_lon is None:
            sample.reward = 0.0
            sample.reward_details = {"error": "missing_latlon"}
            continue
        pred_geo = build_geo_point(pred_lat, pred_lon, source="parsed_prediction_latlon")
        gt_lat = sample.gt_latitude
        gt_lon = sample.gt_longitude
        gt_geo = None
        if gt_lat is None or gt_lon is None:
            if geocoder is None or not sample.gt_text:
                sample.reward = 0.0
                sample.reward_details = {"error": "missing_gt_latlon"}
                continue
            gt_geo_result = geocoder.geocode(sample.gt_text)
            if not gt_geo_result or gt_geo_result.get("status") != "success":
                sample.reward = 0.0
                sample.reward_details = {"error": "gt_geocode_failed"}
                continue
            gt_lat = gt_geo_result["latitude"]
            gt_lon = gt_geo_result["longitude"]
            gt_geo = build_geo_point(
                gt_lat,
                gt_lon,
                source="geocoded_ground_truth_text",
                raw_geo=gt_geo_result,
            )
        else:
            gt_geo = build_geo_point(gt_lat, gt_lon, source="annotation_latlon")
        distance = geodesic_km(pred_lat, pred_lon, gt_lat, gt_lon)
        sample.reward = calculate_geoscore(distance, max_distance)
        sample.reward_details = build_geoscore_details(pred_geo, gt_geo, distance, sample.reward)


def run_llm_judge(
    args: argparse.Namespace,
    samples: List[SampleResult],
    judge_prompt_template: str,
) -> None:
    if not args.judge_model:
        raise ValueError("--judge-model is required for reward-mode=llm_judge")
    ensure_repo_on_path()
    from swift.infer_engine import RequestConfig, VllmEngine

    judge_model_path = resolve_model_path(args.judge_model)
    judge_profile = detect_model_profile(judge_model_path)
    template = build_infer_template(judge_profile, args.disable_model_thinking)
    engine_kwargs = build_vllm_engine_kwargs(args, judge_profile)
    engine = VllmEngine(
        judge_profile["model_path"],
        template=template,
        model_type=judge_profile.get("swift_model_type"),
        template_type=judge_profile.get("swift_template_type"),
        **engine_kwargs,
    )
    request_config = RequestConfig(
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        top_p=1.0,
    )
    batched_samples = list(chunked(samples, args.infer_batch_size))
    total_judge_samples = sum(1 for sample in samples if sample.gt_text and sample.parsed_answer)
    with tqdm(total=total_judge_samples, desc="Scoring llm_judge", unit="sample") as pbar:
        for batch in batched_samples:
            payloads = []
            for sample in batch:
                prediction = sample.parsed_answer
                ground_truth = sample.gt_text
                if not ground_truth:
                    sample.reward = 0.0
                    sample.reward_details = {"error": "missing_gt_text"}
                    continue
                if not prediction:
                    sample.reward = 0.0
                    sample.reward_details = {"error": "missing_parsed_answer"}
                    continue
                payloads.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": build_judge_prompt(
                                    judge_prompt_template,
                                    prediction,
                                    ground_truth,
                                    args.max_distance,
                                ),
                            }
                        ]
                    }
                )
            if not payloads:
                continue
            responses = engine.infer(payloads, request_config=request_config)
            response_iter = iter(responses)
            for sample in batch:
                if sample.reward_details.get("error") in {"missing_gt_text", "missing_parsed_answer"}:
                    continue
                response = next(response_iter)
                raw_output = response.choices[0].message.content if not isinstance(response, Exception) else ""
                parsed = extract_last_valid_judge_json(raw_output)
                if parsed is None:
                    sample.reward = 0.0
                    sample.reward_details = {
                        "judge_output": raw_output,
                        "judge_json": None,
                        "error": "judge_json_parse_failed",
                        "output_truncated": bool(raw_output) and not raw_output.rstrip().endswith("}"),
                    }
                    continue
                score = maybe_float(parsed.get("score"))
                sample.reward = min(1.0, max(0.0, score if score is not None else 0.0))
                sample.reward_details = {
                    "judge_output": raw_output,
                    "judge_json": parsed,
                    "output_truncated": bool(raw_output) and not raw_output.rstrip().endswith("}"),
                }
            pbar.update(len(payloads))
    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def compute_probabilities(
    items: Sequence[DatasetItem],
    samples: Sequence[SampleResult],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rewards_by_item: Dict[str, List[SampleResult]] = {item.item_id: [] for item in items}
    for sample in samples:
        rewards_by_item[sample.item_id].append(sample)

    reward_means = []
    reward_stds = []
    image_stats: List[Dict[str, Any]] = []
    for item in tqdm(items, desc="Aggregating image stats", unit="image"):
        item_samples = rewards_by_item[item.item_id]
        reward_values = np.array([sample.reward for sample in item_samples], dtype=np.float64)
        mu = float(np.mean(reward_values)) if len(reward_values) else 0.0
        sigma = float(np.std(reward_values)) if len(reward_values) else 0.0
        reward_means.append(mu)
        reward_stds.append(sigma)
        image_stats.append(
            {
                "item_id": item.item_id,
                "image_path": item.image_path,
                "relative_path": item.relative_path,
                "gt_text": item.gt_text,
                "gt_latitude": item.gt_latitude,
                "gt_longitude": item.gt_longitude,
                "num_samples": int(len(item_samples)),
                "reward_mean": mu,
                "reward_std": sigma,
                "reward_min": float(np.min(reward_values)) if len(reward_values) else 0.0,
                "reward_max": float(np.max(reward_values)) if len(reward_values) else 0.0,
                "sample_rewards": [sample.reward for sample in item_samples],
            }
        )

    mu_arr = np.array(reward_means, dtype=np.float64)
    sigma_arr = np.array(reward_stds, dtype=np.float64)
    m_mu = float(args.mid_target if args.mid_target is not None else np.mean(mu_arr))
    s_mu = float(np.std(mu_arr))
    s_sigma = float(np.std(sigma_arr))

    if args.variance_mode == "percentile":
        order = np.argsort(np.argsort(sigma_arr))
        percentile = order / max(1, len(sigma_arr) - 1)
    else:
        percentile = None

    values = []
    for idx, stat in enumerate(image_stats):
        mu = stat["reward_mean"]
        sigma = stat["reward_std"]
        s_mid = math.exp(-((mu - m_mu) ** 2) / (2 * (args.tau_mu * s_mu) ** 2 + args.epsilon))
        if args.variance_mode == "exp_std":
            s_var = 1 - math.exp(-(sigma) / (args.tau_sigma * s_sigma + args.epsilon))
        else:
            s_var = float(percentile[idx])
        value = s_mid * s_var
        stat["s_mid"] = float(s_mid)
        stat["s_var"] = float(s_var)
        stat["value"] = float(value)
        values.append(value)

    values_arr = np.array(values, dtype=np.float64)
    if values_arr.sum() <= 0:
        probs = np.full_like(values_arr, fill_value=1.0 / max(1, len(values_arr)))
    else:
        probs = values_arr / values_arr.sum()
    for stat, prob in zip(image_stats, probs):
        stat["probability"] = float(prob)
    return image_stats


def weighted_sample_without_replacement(
    stats: Sequence[Dict[str, Any]],
    keep_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if keep_count >= len(stats):
        return list(stats)
    rng = random.Random(seed)
    keys = []
    for stat in stats:
        prob = max(float(stat["probability"]), 1e-12)
        # Gumbel-top-k: score = log(weight) + Gumbel(0,1), then keep top-k.
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        gumbel = -math.log(-math.log(u))
        score = math.log(prob) + gumbel
        keys.append((score, stat))
    keys.sort(key=lambda x: x[0], reverse=True)
    return [stat for _, stat in keys[:keep_count]]


def materialize_selection(
    stats: Sequence[Dict[str, Any]],
    output_dir: Path,
    selected_root_name: str,
    export_mode: str,
) -> None:
    if export_mode == "manifest_only":
        return
    selected_root = output_dir / selected_root_name
    for stat in tqdm(stats, desc=f"Exporting ({export_mode})", unit="image"):
        src = Path(stat["image_path"])
        dst = selected_root / stat["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if export_mode == "symlink":
            os.symlink(src, dst)
        elif export_mode == "copy":
            shutil.copy2(src, dst)


def save_run_outputs(
    args: argparse.Namespace,
    items: Sequence[DatasetItem],
    samples: Sequence[SampleResult],
    image_stats: Sequence[Dict[str, Any]],
    selected_stats: Sequence[Dict[str, Any]],
    model_profiles: Sequence[Dict[str, Any]],
    judge_profile: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(vars(args))
    run_config["generation_model_profiles"] = list(model_profiles)
    if judge_profile is not None:
        run_config["judge_model_profile"] = judge_profile
    write_json(output_dir / "run_config.json", run_config)
    write_jsonl(output_dir / "discovered_items.jsonl", [asdict(item) for item in items])
    write_jsonl(output_dir / "samples.jsonl", [asdict(sample) for sample in samples])
    write_jsonl(output_dir / "image_stats.jsonl", image_stats)
    selected_ids = {stat["item_id"] for stat in selected_stats}
    for stat in image_stats:
        stat["selected"] = stat["item_id"] in selected_ids
    write_jsonl(output_dir / "selection.jsonl", image_stats)
    write_json(output_dir / "summary.json", {
        "num_items": len(items),
        "num_samples": len(samples),
        "num_selected": len(selected_stats),
        "reward_mode": args.reward_mode,
        "variance_mode": args.variance_mode,
        "export_mode": args.export_mode,
        "generation_model_profiles": list(model_profiles),
        "judge_model_profile": judge_profile,
    })


def allocate_samples(models: List[str], args: argparse.Namespace) -> List[int]:
    if args.samples_per_model:
        counts = [int(x) for x in parse_csv_list(args.samples_per_model)]
        if len(counts) != len(models):
            raise ValueError("--samples-per-model must have the same length as --models")
        return counts
    base = args.total_samples // len(models)
    rem = args.total_samples % len(models)
    counts = [base] * len(models)
    for idx in range(rem):
        counts[idx] += 1
    return counts


def maybe_make_geocoder(args: argparse.Namespace, output_dir: Path) -> Optional[OpenCageWrapper]:
    api_keys = parse_csv_list(args.opencage_api_keys)
    if not api_keys:
        return None
    cache_file = Path(args.geocode_cache_file) if args.geocode_cache_file else output_dir / "geocode_cache.json"
    cache = OpenCageCache(cache_file)
    return OpenCageWrapper(api_keys, cache, retry_error_cache=args.retry_error_cache)


def save_model_samples(output_dir: Path, model_name: str, samples: Sequence[SampleResult]) -> None:
    write_jsonl(output_dir / "generations" / f"{sanitize_name(model_name)}.jsonl", [asdict(sample) for sample in samples])


def load_model_samples(output_dir: Path, model_name: str) -> Optional[List[SampleResult]]:
    path = output_dir / "generations" / f"{sanitize_name(model_name)}.jsonl"
    if not path.exists():
        return None
    try:
        rows = read_jsonl(path)
        return [SampleResult(**row) for row in rows]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def load_complete_model_samples(
    output_dir: Path,
    model_name: str,
    run_model_name: str,
    items: Sequence[DatasetItem],
    samples_per_image: int,
    reward_mode: str,
) -> Optional[List[SampleResult]]:
    saved_samples = load_model_samples(output_dir, model_name)
    if saved_samples is None:
        return None
    expected_total = len(items) * samples_per_image
    if len(saved_samples) != expected_total:
        return None

    expected_item_ids = {item.item_id for item in items}
    item_sample_indices: Dict[str, set[int]] = {item_id: set() for item_id in expected_item_ids}
    for sample in saved_samples:
        if sample.model_name != run_model_name or sample.reward_mode != reward_mode:
            return None
        if sample.item_id not in item_sample_indices:
            return None
        if not 0 <= sample.sample_index < samples_per_image:
            return None
        if sample.sample_index in item_sample_indices[sample.item_id]:
            return None
        item_sample_indices[sample.item_id].add(sample.sample_index)

    if any(len(sample_indices) != samples_per_image for sample_indices in item_sample_indices.values()):
        return None
    return saved_samples


def main() -> None:
    args = parse_args()
    validate_parallel_config(args)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    set_runtime_env(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = build_dataset(args)
    prompt_file = Path(args.prompt_file) if args.prompt_file else default_generation_prompt_file(args.reward_mode)
    system_prompt = read_text(prompt_file)

    model_names = parse_csv_list(args.models)
    model_profiles = [detect_model_profile(resolve_model_path(model_name)) for model_name in model_names]
    sample_counts = allocate_samples(model_names, args)

    all_samples: List[SampleResult] = []
    for model_name, model_profile, sample_count in tqdm(
        list(zip(model_names, model_profiles, sample_counts)), desc="Models", unit="model"
    ):
        if sample_count <= 0:
            continue
        run_model_name = sanitize_name(Path(model_name).name)
        cached_samples = load_complete_model_samples(
            output_dir=output_dir,
            model_name=model_name,
            run_model_name=run_model_name,
            items=items,
            samples_per_image=sample_count,
            reward_mode=args.reward_mode,
        )
        if cached_samples is not None:
            print(
                f"Reusing completed generations for model {model_name} "
                f"from {output_dir / 'generations' / f'{sanitize_name(model_name)}.jsonl'}"
            )
            all_samples.extend(cached_samples)
            continue
        model_system_prompt = adapt_prompt_for_model(system_prompt, model_profile)
        results = run_generation_for_model(
            args=args,
            model_profile=model_profile,
            model_name=run_model_name,
            items=items,
            system_prompt=model_system_prompt,
            samples_per_image=sample_count,
        )
        save_model_samples(output_dir, model_name, results)
        all_samples.extend(results)

    geocoder: Optional[OpenCageWrapper] = None
    if args.reward_mode in {"text_geocode", "latlon_geoscore"}:
        geocoder = maybe_make_geocoder(args, output_dir)
        if args.reward_mode == "text_geocode" and geocoder is None:
            raise ValueError("--opencage-api-keys is required for reward-mode=text_geocode")
        score_text_geocode(all_samples, geocoder, args.max_distance, args.confidence_threshold) if args.reward_mode == "text_geocode" else score_latlon_geoscore(all_samples, geocoder, args.max_distance)
        if geocoder is not None:
            geocoder.cache.flush()
    else:
        run_llm_judge(
            args,
            all_samples,
            judge_prompt_template=read_text(default_judge_prompt_file(args.judge_prompt_mode)),
        )

    judge_profile = None
    if args.reward_mode == "llm_judge" and args.judge_model:
        judge_profile = detect_model_profile(resolve_model_path(args.judge_model))

    image_stats = compute_probabilities(items, all_samples, args)
    keep_count = args.keep_count if args.keep_count is not None else len(image_stats)
    print(f"Selecting {keep_count} / {len(image_stats)} images")
    selected_stats = weighted_sample_without_replacement(image_stats, keep_count, args.random_seed)
    materialize_selection(selected_stats, output_dir, args.selected_root_name, args.export_mode)
    save_run_outputs(args, items, all_samples, image_stats, selected_stats, model_profiles, judge_profile)


if __name__ == "__main__":
    main()
