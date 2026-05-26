#!/usr/bin/env python3
"""Run VLM inference on JSON image records and write answers back to JSON."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


DEFAULT_MODEL_ROOT = Path("/nfs/sunboyuan/model")
DEFAULT_REPO_ROOT = Path("/nfs/sunboyuan/Geobench/ms-swift")

SUPPORTED_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Qwen3.5-9B": ["Qwen3.5-9B"],
    "Qwen3.5-27B": ["Qwen3.5-27B"],
    "InternVL3.5-8B": ["InternVL3_5-8B", "InternVL3.5-8B"],
    "InternVL3.5-14B": ["InternVL3_5-14B", "InternVL3.5-14B"],
    "Qwen3-VL-8B": ["Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"],
    "Penguin-VL-8B": ["Penguin-VL-8B"],
    "gemma3-12b-it": ["gemma-3-12b-it", "gemma3-12b-it"],
    "MiniCPM-V-4_5": ["MiniCPM-V-4_5", "MiniCPM-V-4.5"],
}


class BatchTimeoutError(TimeoutError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM inference for a JSON list with image paths.")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODEL_CANDIDATES.keys()))
    parser.add_argument(
        "--model-path",
        help="Optional direct model path. If set, it overrides --model path resolution.",
    )
    parser.add_argument("--input-json", required=True, help="Input JSON file. Must be a list of dict.")
    parser.add_argument("--prompt-file", required=True, help="TXT prompt file used as system prompt.")
    parser.add_argument(
        "--output-json",
        help="Output JSON path. Default: <input_stem>_<model_field>_out.json in same directory.",
    )
    parser.add_argument(
        "--image-field",
        default="image",
        help="Field name containing image path in each JSON item. Default: image",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Optional root for relative image paths. Default: input-json parent directory.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-timeout-sec",
        type=float,
        default=300.0,
        help="Per-batch timeout in seconds. <=0 means no timeout.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--disable-model-thinking",
        action="store_true",
        help="Disable native thinking mode for supported templates.",
    )
    parser.add_argument(
        "--log-file",
        help="Log file path. Default: <output_json_stem>.log",
    )
    return parser.parse_args()


def ensure_repo_on_path() -> None:
    if str(DEFAULT_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_REPO_ROOT))


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


def resolve_model_path(model_name: str, explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"--model-path does not exist: {explicit_path}")
        model_root = find_model_root(path)
        if model_root is None:
            raise FileNotFoundError(f"Cannot find config.json under --model-path: {explicit_path}")
        return str(model_root.resolve())

    for candidate_name in SUPPORTED_MODEL_CANDIDATES[model_name]:
        candidate = DEFAULT_MODEL_ROOT / candidate_name
        if not candidate.exists():
            continue
        model_root = find_model_root(candidate)
        if model_root is not None:
            return str(model_root.resolve())
    tried = ", ".join(SUPPORTED_MODEL_CANDIDATES[model_name])
    raise FileNotFoundError(
        f"Cannot resolve model '{model_name}' under {DEFAULT_MODEL_ROOT}. Tried: {tried}. "
        "You can pass --model-path explicitly."
    )


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
    elif "minicpm" in model_type or any("minicpm" in x for x in architectures):
        family = "minicpm"

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
    elif family == "minicpm":
        swift_model_type = "minicpmv"
        swift_template_type = "minicpmv"

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
    }


def should_strip_think_tokens(model_profile: Dict[str, Any]) -> bool:
    swift_model_type = str(model_profile.get("swift_model_type") or "").lower()
    swift_template_type = str(model_profile.get("swift_template_type") or "").lower()
    model_dir_name = str(model_profile.get("model_dir_name") or "").lower()
    model_path = str(model_profile.get("model_path") or "").lower()

    markers = (
        "internvl3_5",
        "internvl3.5",
        "intern3_5vl",
        "intern3.5vl",
    )
    combined = " ".join((swift_model_type, swift_template_type, model_dir_name, model_path))
    return any(marker in combined for marker in markers)


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


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("vlm_json_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def read_input_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item at index {i} is not a dict.")
        rows.append(row)
    return rows


def build_field_prefix(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_image_path(raw_path: str, image_root: Path) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    return str((image_root / p).resolve())


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


@contextmanager
def batch_timeout(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # type: ignore[unused-argument]
        raise BatchTimeoutError(f"Batch exceeded timeout: {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_inference(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], bool, str, str]:
    ensure_repo_on_path()
    from swift.infer_engine import RequestConfig, VllmEngine

    model_path = resolve_model_path(args.model, args.model_path)
    model_profile = detect_model_profile(model_path)
    if not model_profile.get("supports_images", False):
        raise ValueError(f"Model is not detected as multimodal image model: {model_path}")

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    prompt_text = adapt_prompt_for_model(prompt_text, model_profile)

    template = build_infer_template(model_profile, args.disable_model_thinking)
    engine_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }
    if model_profile.get("limit_mm_per_prompt"):
        engine_kwargs["limit_mm_per_prompt"] = model_profile["limit_mm_per_prompt"]

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

    prefix = build_field_prefix(args.model)
    answer_field = f"{prefix}_answer"
    completed_field = f"{prefix}_inference_completed"

    input_json = Path(args.input_json).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else input_json.parent

    for row in rows:
        row.setdefault(answer_field, "")
        row.setdefault(completed_field, False)

    interrupted = False
    with tqdm(total=len(rows), desc="Inferring", unit="item") as pbar:
        for batch_indices in chunked(list(range(len(rows))), args.batch_size):
            payloads = []
            valid_indices: List[int] = []
            for idx in batch_indices:
                row = rows[idx]
                raw_image = row.get(args.image_field)
                if not raw_image:
                    row[completed_field] = False
                    row[answer_field] = ""
                    logger.warning("Missing image field '%s' at item index=%d", args.image_field, idx)
                    continue
                image_path = resolve_image_path(str(raw_image), image_root)
                payloads.append(
                    {
                        "messages": [
                            {"role": "system", "content": prompt_text},
                            {"role": "user", "content": "Locate this image."},
                        ],
                        "images": [image_path],
                    }
                )
                valid_indices.append(idx)

            if not payloads:
                pbar.update(len(batch_indices))
                continue

            try:
                with batch_timeout(args.batch_timeout_sec):
                    responses = engine.infer(payloads, request_config=request_config)
            except BatchTimeoutError:
                logger.error(
                    "Batch timeout triggered after %.2fs at batch starting index=%d. Inference interrupted.",
                    args.batch_timeout_sec,
                    batch_indices[0],
                )
                interrupted = True
                break
            except Exception as exc:
                logger.exception("Batch failed at index=%d with error: %s", batch_indices[0], exc)
                interrupted = True
                break

            for response, idx in zip(responses, valid_indices):
                if isinstance(response, Exception):
                    rows[idx][answer_field] = ""
                    rows[idx][completed_field] = False
                    logger.warning("Item index=%d returned exception response: %s", idx, response)
                    continue
                try:
                    content = response.choices[0].message.content
                except Exception:
                    content = ""
                rows[idx][answer_field] = content if content is not None else ""
                rows[idx][completed_field] = True
            pbar.update(len(batch_indices))

    if interrupted:
        for row in rows:
            if not row.get(completed_field, False):
                row[completed_field] = False

    del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return rows, (not interrupted), answer_field, completed_field


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    input_json = Path(args.input_json).resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    if not Path(args.prompt_file).exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    field_prefix = build_field_prefix(args.model)
    default_output = input_json.with_name(f"{input_json.stem}_{field_prefix}_out.json")
    output_json = Path(args.output_json).resolve() if args.output_json else default_output
    output_json.parent.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file).resolve() if args.log_file else output_json.with_suffix(".log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file)

    logger.info("Start inference")
    logger.info("model=%s", args.model)
    logger.info("gpu=%s", args.gpu)
    logger.info("input_json=%s", input_json)
    logger.info("prompt_file=%s", Path(args.prompt_file).resolve())
    logger.info("output_json=%s", output_json)
    logger.info("batch_size=%d batch_timeout_sec=%.2f", args.batch_size, args.batch_timeout_sec)

    rows = read_input_json(input_json)
    rows, run_completed, answer_field, completed_field = run_inference(rows, args, logger)

    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished. run_completed=%s", run_completed)
    logger.info("answer_field=%s completed_field=%s", answer_field, completed_field)
    logger.info("Wrote %d rows to %s", len(rows), output_json)


if __name__ == "__main__":
    main()
