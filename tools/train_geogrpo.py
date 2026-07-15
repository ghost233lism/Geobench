#!/usr/bin/env python3
"""One-stop GeoBench GRPO data scheduler and ms-swift launcher."""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from statistics import median


DEFAULT_INPUT = Path('/nfs/sunboyuan/Geobench/dataset/data_train/all/all_selected_merged.jsonl')
DEFAULT_MODEL = Path('/nfs/sunboyuan/model/Qwen3.5-2B')
DEFAULT_OUTPUT_ROOT = Path('/nfs/sunboyuan/Geobench/dataset/data_train/all/scheduled')
DEFAULT_SYSTEM_PROMPT_FILE = Path('/nfs/sunboyuan/Geobench/ms-swift/tools/system_prompt.txt')
DEFAULT_CUDA_VISIBLE_DEVICES = '0,1'
DEFAULT_NPROC_PER_NODE = '2'
DEFAULT_IMAGE_MAX_TOKEN_NUM = 'none'
DEFAULT_DOMAIN_RATIO = {
    'ground': 3,
    'map': 3,
    'remote': 3,
    'street': 3,
    'indoor': 1,
    'landmark': 1,
    'roadnet': 1,
    'shape': 1,
    'space': 1,
    'uav': 1,
}
BUCKET_ORDER = [
    'high_mean_low_std',
    'low_mean_low_std',
    'high_mean_high_std',
    'low_mean_high_std',
]
USER_PROMPT = '<image> Based on the image, tell me the specific location and your thinking process. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'
NO_TAGS_USER_PROMPT = '<image> Based on the image, reason carefully about the visual clues and tell me the specific location. Output only the required JSON.'


def parse_train_data_ratio(value):
    if isinstance(value, (float, int)):
        ratio = float(value)
    else:
        text = str(value).strip().lower()
        if text in ('', 'all'):
            return 1.0
        try:
            if text.endswith('%'):
                ratio = float(text[:-1]) / 100.0
            else:
                ratio = float(text)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                '--train-data-ratio must be a float in (0, 1] or a percentage like 25%.') from e
    if not 0.0 < ratio <= 1.0:
        raise argparse.ArgumentTypeError('--train-data-ratio must be > 0 and <= 1.')
    return ratio


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare scheduled GeoBench GRPO data and launch swift rlhf.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--model-type', default=None)
    parser.add_argument('--system-prompt-file', type=Path, default=DEFAULT_SYSTEM_PROMPT_FILE)
    parser.add_argument('--user-prompt', default=USER_PROMPT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--schedule', choices=['four_stage', 'curriculum_mix', 'random'], default='four_stage')
    parser.add_argument('--domain-balance', choices=['ratio', 'random', 'sequential_domain'], default='ratio')
    parser.add_argument('--domain-ratio', default=None, help='Example: ground:2,street:2,indoor:1,uav:1')
    parser.add_argument('--domain-order', default=None, help='Comma or whitespace separated domain order prefix.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--train-data-ratio',
        type=parse_train_data_ratio,
        default=1.0,
        help='Randomly keep this fraction of input data before bucket sorting and domain scheduling.')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cuda-visible-devices', default=DEFAULT_CUDA_VISIBLE_DEVICES)
    parser.add_argument('--nproc-per-node', default=DEFAULT_NPROC_PER_NODE)
    parser.add_argument('--image-max-token-num', default=DEFAULT_IMAGE_MAX_TOKEN_NUM)
    parser.add_argument('--output-dir', type=Path, default=None, help='Swift checkpoint output directory.')
    parser.add_argument('--num-train-epochs', default='1')
    parser.add_argument('--num-generations', default='8')
    parser.add_argument('--per-device-train-batch-size', default='1')
    parser.add_argument('--gradient-accumulation-steps', default='8')
    parser.add_argument('--learning-rate', default='1e-5')
    parser.add_argument('--torch-dtype', default='bfloat16')
    parser.add_argument('--max-length', default='4096')
    parser.add_argument('--max-completion-length', default='1024')
    parser.add_argument('--max-pixels', default=None, help='Optional Swift max_pixels limit. Omit for no limit.')
    parser.add_argument('--deepspeed', default='zero3')
    parser.add_argument('--vllm-mode', choices=['colocate', 'server'], default='colocate')
    parser.add_argument('--vllm-gpu-memory-utilization', default='0.7')
    parser.add_argument('--vllm-tensor-parallel-size', default='1')
    parser.add_argument('--vllm-server-host', nargs='+', default=['127.0.0.1'])
    parser.add_argument('--vllm-server-port', nargs='+', default=['8000'])
    parser.add_argument('--vllm-server-base-url', nargs='+', default=None)
    parser.add_argument('--vllm-server-timeout', default='240')
    parser.add_argument('--vllm-server-group-port', nargs='+', default=None)
    parser.add_argument('--vllm-server-pass-dataset', default=None)
    parser.add_argument('--enable-thinking', default='false')
    parser.add_argument('--response-prefix', default='')
    parser.add_argument('--move-model-batches', default='8')
    parser.add_argument('--sleep-level', default='1')
    parser.add_argument('--offload-optimizer', default='false')
    parser.add_argument('--offload-model', default='false')
    parser.add_argument('--save-strategy', default='steps')
    parser.add_argument('--save-steps', default='100')
    parser.add_argument('--save-total-limit', default='200')
    parser.add_argument('--logging-steps', default='1')
    parser.add_argument('--warmup-ratio', default='0.01')
    parser.add_argument('--max-grad-norm', default='1.0')
    parser.add_argument('--dataloader-num-workers', default='8')
    parser.add_argument('--dataset-num-proc', default='8')
    parser.add_argument('--temperature', default='0.7')
    parser.add_argument('--gradient-checkpointing', default='true')
    parser.add_argument('--vit-gradient-checkpointing', default='false')
    parser.add_argument('--freeze-vit', default='true')
    parser.add_argument('--freeze-aligner', default='true')
    parser.add_argument('--beta', default='0.001')
    parser.add_argument('--num-iterations', default='1')
    parser.add_argument('--swanlab-project', default='geolocation')
    parser.add_argument('--swanlab-exp-name', default=None)
    parser.add_argument('--reward-funcs', nargs='+', default=['geoscore_accuracy', 'geo_format'])
    parser.add_argument('--reward-weights', nargs='+', default=['1.5', '1.0'])
    parser.add_argument('--geoscore-api-keys', nargs='+', default=None)
    parser.add_argument('--geoscore-max-distance', default='2000.0')
    parser.add_argument('--extra-swift-args', nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def parse_domain_ratio(text):
    if not text:
        return dict(DEFAULT_DOMAIN_RATIO)
    ratio = {}
    for item in text.split(','):
        if not item.strip():
            continue
        if ':' not in item:
            raise ValueError(f'Invalid --domain-ratio item: {item}')
        domain, value = item.split(':', 1)
        value = int(value)
        if value <= 0:
            raise ValueError(f'Domain ratio must be positive: {item}')
        ratio[domain.strip()] = value
    if not ratio:
        raise ValueError('--domain-ratio produced an empty ratio.')
    return ratio


def parse_domain_order(text):
    if not text:
        return []
    return [item.strip() for item in text.replace(',', ' ').split() if item.strip()]


def load_rows(path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ('id', 'path', 'latitude', 'longitude', 'reward_mean', 'reward_std', 'category'):
                if key not in row:
                    raise ValueError(f'Missing `{key}` in {path}:{line_no}')
            rows.append(row)
    return rows


def sample_rows_by_ratio(rows, ratio, rng):
    if not rows or ratio >= 1.0:
        return rows
    sample_size = max(1, int(len(rows) * ratio))
    if sample_size >= len(rows):
        return rows
    return rng.sample(rows, sample_size)


def assign_buckets(rows):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row['category']].append(row)

    medians = {}
    for domain, domain_rows in by_domain.items():
        mean_mid = median(float(row['reward_mean']) for row in domain_rows)
        std_mid = median(float(row['reward_std']) for row in domain_rows)
        medians[domain] = {'reward_mean_median': mean_mid, 'reward_std_median': std_mid}
        for row in domain_rows:
            mean_level = 'high_mean' if float(row['reward_mean']) >= mean_mid else 'low_mean'
            std_level = 'high_std' if float(row['reward_std']) >= std_mid else 'low_std'
            row['bucket'] = f'{mean_level}_{std_level}'
    return by_domain, medians


def order_domain_rows(domain_rows, schedule, rng):
    buckets = {bucket: [] for bucket in BUCKET_ORDER}
    for row in domain_rows:
        buckets[row['bucket']].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    if schedule == 'random':
        ordered = list(domain_rows)
        rng.shuffle(ordered)
        return ordered

    if schedule == 'four_stage':
        ordered = []
        for bucket in BUCKET_ORDER:
            ordered.extend(buckets[bucket])
        return ordered

    return curriculum_mix_order(buckets, rng)


def curriculum_mix_order(buckets, rng):
    bucket_queues = {name: deque(values) for name, values in buckets.items()}
    total = sum(len(values) for values in buckets.values())
    phases = [
        {'high_mean_low_std': 0.70, 'low_mean_low_std': 0.20, 'high_mean_high_std': 0.10, 'low_mean_high_std': 0.00},
        {'high_mean_low_std': 0.30, 'low_mean_low_std': 0.35, 'high_mean_high_std': 0.25, 'low_mean_high_std': 0.10},
        {'high_mean_low_std': 0.15, 'low_mean_low_std': 0.25, 'high_mean_high_std': 0.35, 'low_mean_high_std': 0.25},
    ]
    ordered = []
    phase_size = max(1, total // len(phases))
    for phase in phases:
        target = min(phase_size, total - len(ordered))
        weights = [phase[name] for name in BUCKET_ORDER]
        while target > 0 and any(bucket_queues.values()):
            bucket = rng.choices(BUCKET_ORDER, weights=weights, k=1)[0]
            if not bucket_queues[bucket]:
                available = [name for name in BUCKET_ORDER if bucket_queues[name]]
                bucket = rng.choice(available)
            ordered.append(bucket_queues[bucket].popleft())
            target -= 1
    for bucket in BUCKET_ORDER:
        ordered.extend(bucket_queues[bucket])
    return ordered


def ordered_domain_names(ordered_by_domain, domain_order):
    available = {name for name, rows in ordered_by_domain.items() if rows}
    names = []
    seen = set()
    for domain in domain_order:
        if domain in available and domain not in seen:
            names.append(domain)
            seen.add(domain)
    names.extend(domain for domain in sorted(available) if domain not in seen)
    return names


def interleave_domains(ordered_by_domain, domain_balance, domain_ratio, rng, domain_order=None):
    domain_names = ordered_domain_names(ordered_by_domain, domain_order or [])
    if domain_balance == 'sequential_domain':
        rows = []
        for domain in domain_names:
            rows.extend(ordered_by_domain[domain])
        return rows

    queues = {domain: deque(ordered_by_domain[domain]) for domain in domain_names}
    cycle = []
    for domain in domain_names:
        cycle.extend([domain] * domain_ratio.get(domain, 1))
    if not cycle:
        raise ValueError('No domains available for ratio scheduling.')

    rows = []
    if domain_balance == 'random':
        while any(queues.values()):
            active_domains = [domain for domain in domain_names if queues[domain]]
            domain = rng.choice(active_domains)
            rows.append(queues[domain].popleft())
        return rows

    while any(queues.values()):
        made_progress = False
        for domain in cycle:
            if queues.get(domain):
                rows.append(queues[domain].popleft())
                made_progress = True
        if not made_progress:
            remaining = [domain for domain, queue in queues.items() if queue]
            for domain in remaining:
                rows.append(queues[domain].popleft())
    return rows


def load_system_prompt(path):
    try:
        return path.read_text(encoding='utf-8').strip()
    except OSError as e:
        raise RuntimeError(f'Failed to read --system-prompt-file {path}: {e}') from e


def to_swift_row(row, system_prompt, user_prompt):
    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'images': [row['path']],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'id': row['id'],
        'category': row['category'],
        'reward_mean': row['reward_mean'],
        'reward_std': row['reward_std'],
        'bucket': row['bucket'],
    }


def write_jsonl(rows, output_path, system_prompt, user_prompt):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(to_swift_row(row, system_prompt, user_prompt), ensure_ascii=False) + '\n')


def print_stats(rows, medians, domain_ratio):
    print(f'Prepared rows: {len(rows)}')
    print('Domain ratio weights:', ', '.join(f'{k}:{domain_ratio[k]}' for k in sorted(domain_ratio)))
    print('\nDomain counts:')
    domain_counts = Counter(row['category'] for row in rows)
    for domain, count in sorted(domain_counts.items()):
        print(f'  {domain}\t{count}')
    print('\nBucket counts:')
    bucket_counts = Counter((row['category'], row['bucket']) for row in rows)
    for domain in sorted(domain_counts):
        parts = [f'{bucket}={bucket_counts[(domain, bucket)]}' for bucket in BUCKET_ORDER]
        print(f'  {domain}\t' + ', '.join(parts))
    print('\nDomain medians:')
    for domain in sorted(medians):
        values = medians[domain]
        print(
            f"  {domain}\treward_mean={values['reward_mean_median']:.6f}, "
            f"reward_std={values['reward_std_median']:.6f}")


def build_swift_command(args, dataset_path, run_name):
    output_dir = args.output_dir or Path('/nfs/sunboyuan/Geobench/ms-swift/output') / run_name
    cmd = [
        'swift', 'rlhf',
        '--rlhf_type', 'grpo',
        '--model', str(args.model),
        '--reward_funcs', *args.reward_funcs,
        '--reward_weights', *args.reward_weights,
        '--train_type', 'full',
        '--freeze_vit', args.freeze_vit,
        '--freeze_aligner', args.freeze_aligner,
        '--torch_dtype', args.torch_dtype,
        '--use_vllm', 'true',
        '--vllm_mode', args.vllm_mode,
        '--enable_thinking', args.enable_thinking,
        '--response_prefix', args.response_prefix,
        '--offload_optimizer', args.offload_optimizer,
        '--offload_model', args.offload_model,
        '--deepspeed', args.deepspeed,
        '--dataset', str(dataset_path),
        '--remove_unused_columns', 'false',
        '--dataset_shuffle', 'false',
        '--train_dataloader_shuffle', 'false',
        '--load_from_cache_file', 'false',
        '--max_length', args.max_length,
        '--max_completion_length', args.max_completion_length,
        '--num_train_epochs', args.num_train_epochs,
        '--per_device_train_batch_size', args.per_device_train_batch_size,
        '--gradient_accumulation_steps', args.gradient_accumulation_steps,
        '--learning_rate', args.learning_rate,
        '--lr_scheduler_type', 'cosine',
        '--warmup_ratio', args.warmup_ratio,
        '--save_strategy', args.save_strategy,
        '--save_steps', args.save_steps,
        '--save_total_limit', args.save_total_limit,
        '--logging_steps', args.logging_steps,
        '--dataloader_num_workers', args.dataloader_num_workers,
        '--dataset_num_proc', args.dataset_num_proc,
        '--num_generations', args.num_generations,
        '--temperature', args.temperature,
        '--output_dir', str(output_dir),
        '--log_completions', 'true',
        '--report_to', 'swanlab',
        '--swanlab_project', args.swanlab_project,
        '--swanlab_exp_name', args.swanlab_exp_name or run_name,
        '--gradient_checkpointing', args.gradient_checkpointing,
        '--vit_gradient_checkpointing', args.vit_gradient_checkpointing,
        '--max_grad_norm', args.max_grad_norm,
        '--epsilon', '0.2',
        '--epsilon_high', '0.28',
        '--scale_rewards', 'none',
        '--beta', args.beta,
        '--num_iterations', args.num_iterations,
    ]
    if args.vllm_mode == 'server':
        if args.vllm_server_base_url:
            cmd.extend(['--vllm_server_base_url', *args.vllm_server_base_url])
        else:
            cmd.extend(['--vllm_server_host', *args.vllm_server_host])
            cmd.extend(['--vllm_server_port', *args.vllm_server_port])
        cmd.extend(['--vllm_server_timeout', args.vllm_server_timeout])
        if args.vllm_server_group_port:
            cmd.extend(['--vllm_server_group_port', *args.vllm_server_group_port])
        if args.vllm_server_pass_dataset is not None:
            cmd.extend(['--vllm_server_pass_dataset', args.vllm_server_pass_dataset])
    else:
        cmd.extend([
            '--vllm_gpu_memory_utilization', args.vllm_gpu_memory_utilization,
            '--vllm_tensor_parallel_size', args.vllm_tensor_parallel_size,
            '--move_model_batches', args.move_model_batches,
            '--sleep_level', args.sleep_level,
        ])
    if args.max_pixels is not None:
        cmd.extend(['--max_pixels', args.max_pixels])
    if args.geoscore_api_keys:
        cmd.extend(['--geoscore_api_keys', *args.geoscore_api_keys])
    cmd.extend(['--geoscore_max_distance', args.geoscore_max_distance])
    if args.model_type:
        cmd.extend(['--model_type', args.model_type])
    cmd.extend(args.extra_swift_args)
    return cmd


def shell_join(cmd):
    import shlex
    return ' '.join(shlex.quote(part) for part in cmd)


def redact_command(cmd):
    redacted = []
    redacting = False
    for part in cmd:
        if redacting and part.startswith('--'):
            redacting = False
        if part == '--geoscore_api_keys':
            redacted.append(part)
            redacting = True
            continue
        redacted.append('***' if redacting else part)
    return redacted


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    run_name = args.run_name or datetime.now().strftime('geogrpo_%Y%m%d_%H%M%S')
    output_path = args.output_root / f'{run_name}.jsonl'
    domain_ratio = parse_domain_ratio(args.domain_ratio)
    domain_order = parse_domain_order(args.domain_order)
    system_prompt = load_system_prompt(args.system_prompt_file)

    rows = load_rows(args.input)
    input_row_count = len(rows)
    rows = sample_rows_by_ratio(rows, args.train_data_ratio, rng)
    pre_schedule_row_count = len(rows)
    by_domain, medians = assign_buckets(rows)
    effective_domain_ratio = {domain: domain_ratio.get(domain, 1) for domain in sorted(by_domain)}
    ordered_by_domain = {
        domain: order_domain_rows(domain_rows, args.schedule, rng)
        for domain, domain_rows in by_domain.items()
    }
    scheduled_rows = interleave_domains(
        ordered_by_domain, args.domain_balance, effective_domain_ratio, rng, domain_order)
    if args.max_samples is not None:
        scheduled_rows = scheduled_rows[:args.max_samples]

    write_jsonl(scheduled_rows, output_path, system_prompt, args.user_prompt)
    print(f'Input rows: {input_row_count}')
    print(f'Train data ratio: {args.train_data_ratio:g}; rows before scheduling: {pre_schedule_row_count}')
    print_stats(scheduled_rows, medians, effective_domain_ratio)
    print(f'System prompt file: {args.system_prompt_file}')
    print(f'User prompt: {args.user_prompt}')
    print(f'\nScheduled dataset: {output_path}')

    cmd = build_swift_command(args, output_path, run_name)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    env['NPROC_PER_NODE'] = args.nproc_per_node
    if args.image_max_token_num.lower() not in ('', 'none', 'null', 'false', '0'):
        env['IMAGE_MAX_TOKEN_NUM'] = args.image_max_token_num
    else:
        env.pop('IMAGE_MAX_TOKEN_NUM', None)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('PYTORCH_ALLOC_CONF', env['PYTORCH_CUDA_ALLOC_CONF'])

    print('\nEnvironment:')
    for key in (
            'CUDA_VISIBLE_DEVICES', 'NPROC_PER_NODE', 'IMAGE_MAX_TOKEN_NUM', 'GEOSCORE_CACHE_FILE',
            'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF', 'PYTHONPATH', 'GEOBENCH_LOG_SITECUSTOMIZE',
            'GEOBENCH_PATCH_QWEN35_ZERO3_CONV1D', 'GEOBENCH_DISABLE_QWEN35_CAUSAL_CONV1D'):
        print(f'  {key}={env.get(key, "<unset>")}')
    print('\nSwift command:')
    print(shell_join(redact_command(cmd)))

    if args.dry_run:
        print('\nDry run: generated data and skipped training.')
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(cmd, cwd=repo_root, env=env, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
