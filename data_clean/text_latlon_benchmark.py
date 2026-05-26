#!/usr/bin/env python3
"""Benchmark models on text-address to lat/lon prediction."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


DATA_CLEAN_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = DATA_CLEAN_DIR / "prompt_text_address_to_latlon.txt"
DEFAULT_INPUT_FILE = Path("/nfs/sunboyuan/Geobench/dataset/test_lat_lon.json")
DEFAULT_MODEL_ROOT = Path("/nfs/sunboyuan/model")


@dataclass
class AddressItem:
    item_id: int
    original_solution: str
    formatted_address: str
    gt_latitude: float
    gt_longitude: float
    metadata: Dict[str, Any]


@dataclass
class PredictionResult:
    item_id: int
    model_name: str
    formatted_address: str
    gt_latitude: float
    gt_longitude: float
    raw_output: str
    parsed_latitude: Optional[float]
    parsed_longitude: Optional[float]
    distance_km: Optional[float]
    parse_success: bool


def ensure_repo_on_path() -> None:
    repo_root = Path("/nfs/sunboyuan/Geobench/ms-swift")
    repo_root_str = str(repo_root)
    if repo_root_str not in os.sys.path:
        os.sys.path.insert(0, repo_root_str)


def maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip("/")) or "model"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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

    return {
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


def geodesic_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return 6371.0 * c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark text-address to coordinate prediction.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="JSON file with solution/lat/lng fields.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-model predictions and summary.")
    parser.add_argument(
        "--models",
        default="InternVL3_5-8B,gemma-3-12b-it,Qwen3.5-9B,Qwen3-VL-8B-Instruct,Qwen3_8b",
        help="Comma-separated model directories or names under /nfs/sunboyuan/model.",
    )
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="System prompt file.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--disable-model-thinking", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Evaluate at most this many samples.")
    return parser.parse_args()


def set_runtime_env(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_solution_text(solution: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", solution, flags=re.IGNORECASE | re.DOTALL)
    text = match.group(1) if match else solution
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(solution: str) -> str:
    text = extract_solution_text(solution)
    parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    return ", ".join(parts)


def load_items(input_file: Path, max_samples: Optional[int]) -> List[AddressItem]:
    rows = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {input_file}, got {type(rows).__name__}")
    items: List[AddressItem] = []
    for idx, row in enumerate(rows):
        lat = row.get("lat")
        lon = row.get("lng", row.get("lon", row.get("longitude")))
        if lat is None or lon is None:
            continue
        items.append(
            AddressItem(
                item_id=idx,
                original_solution=row.get("solution", ""),
                formatted_address=normalize_address(row.get("solution", "")),
                gt_latitude=float(lat),
                gt_longitude=float(lon),
                metadata={k: v for k, v in row.items() if k not in {"solution", "lat", "lng", "lon", "longitude"}},
            )
        )
        if max_samples is not None and len(items) >= max_samples:
            break
    return items


def make_requests(items: Sequence[AddressItem], system_prompt: str) -> List[Dict[str, Any]]:
    requests_payload: List[Dict[str, Any]] = []
    for item in items:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Address: {item.formatted_address}"},
        ]
        requests_payload.append({"messages": messages})
    return requests_payload


def summarize_distances(distances: Sequence[float]) -> Dict[str, Optional[float]]:
    if not distances:
        return {
            "mean_distance_km": None,
            "median_distance_km": None,
            "p90_distance_km": None,
        }
    ordered = sorted(float(x) for x in distances)
    p90_idx = min(len(ordered) - 1, max(0, int(0.9 * len(ordered)) - 1))
    return {
        "mean_distance_km": sum(ordered) / len(ordered),
        "median_distance_km": median(ordered),
        "p90_distance_km": ordered[p90_idx],
    }


def run_model(
    args: argparse.Namespace,
    model_name: str,
    items: Sequence[AddressItem],
    system_prompt: str,
) -> Tuple[List[PredictionResult], Dict[str, Any], Dict[str, Any]]:
    ensure_repo_on_path()
    from swift.infer_engine import RequestConfig, VllmEngine

    model_path = resolve_model_path(model_name)
    model_profile = detect_model_profile(model_path)
    template = build_infer_template(model_profile, args.disable_model_thinking)
    engine_kwargs = {
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

    requests_payload = make_requests(items, system_prompt)
    results: List[PredictionResult] = []
    for batch_items, batch_payloads in zip(
        chunked(list(items), args.infer_batch_size),
        chunked(requests_payload, args.infer_batch_size),
    ):
        responses = engine.infer(list(batch_payloads), request_config=request_config)
        for item, response in zip(batch_items, responses):
            raw_output = response.choices[0].message.content if not isinstance(response, Exception) else str(response)
            pred_lat, pred_lon = extract_latlon(raw_output)
            parse_success = pred_lat is not None and pred_lon is not None
            distance_km = (
                geodesic_km(pred_lat, pred_lon, item.gt_latitude, item.gt_longitude)
                if parse_success
                else None
            )
            results.append(
                PredictionResult(
                    item_id=item.item_id,
                    model_name=sanitize_name(Path(model_name).name),
                    formatted_address=item.formatted_address,
                    gt_latitude=item.gt_latitude,
                    gt_longitude=item.gt_longitude,
                    raw_output=raw_output,
                    parsed_latitude=pred_lat,
                    parsed_longitude=pred_lon,
                    distance_km=distance_km,
                    parse_success=parse_success,
                )
            )

    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    distances = [row.distance_km for row in results if row.distance_km is not None]
    summary = {
        "model_name": sanitize_name(Path(model_name).name),
        "model_profile": model_profile,
        "num_samples": len(results),
        "num_parsed": sum(1 for row in results if row.parse_success),
        "parse_success_rate": (sum(1 for row in results if row.parse_success) / len(results)) if results else 0.0,
        **summarize_distances(distances),
    }
    return results, summary, model_profile


def main() -> None:
    args = parse_args()
    set_runtime_env(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(args.input_file)
    items = load_items(input_file, args.max_samples)
    system_prompt = read_text(Path(args.prompt_file))
    model_names = parse_csv_list(args.models)

    write_jsonl(output_dir / "benchmark_items.jsonl", [asdict(item) for item in items])

    summaries: List[Dict[str, Any]] = []
    model_profiles: List[Dict[str, Any]] = []
    for model_name in tqdm(model_names, desc="Models", unit="model"):
        results, summary, model_profile = run_model(args, model_name, items, system_prompt)
        write_jsonl(
            output_dir / "predictions" / f"{sanitize_name(Path(model_name).name)}.jsonl",
            [asdict(row) for row in results],
        )
        summaries.append(summary)
        model_profiles.append(model_profile)

    write_json(
        output_dir / "summary.json",
        {
            "input_file": str(input_file),
            "prompt_file": str(Path(args.prompt_file).resolve()),
            "num_items": len(items),
            "models": model_names,
            "model_profiles": model_profiles,
            "summaries": summaries,
            "config": vars(args),
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
