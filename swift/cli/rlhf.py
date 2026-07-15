# Copyright (c) ModelScope Contributors. All rights reserved.

import importlib.util
import os
import sys
from pathlib import Path


def _load_geobench_runtime_sitecustomize():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_dir = repo_root / 'tools' / 'runtime'
    sitecustomize_path = runtime_dir / 'sitecustomize.py'
    if not sitecustomize_path.exists():
        return
    runtime_dir_text = str(runtime_dir)
    if runtime_dir_text not in sys.path:
        sys.path.insert(0, runtime_dir_text)
    module_name = '_geobench_runtime_sitecustomize'
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, sitecustomize_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if os.environ.get('GEOBENCH_LOG_SITECUSTOMIZE', '').lower() in {'1', 'true', 'yes', 'on'}:
        rank = os.environ.get('RANK', '?')
        local_rank = os.environ.get('LOCAL_RANK', '?')
        print(
            f'[geobench-sitecustomize] explicit load in rlhf.py rank={rank} local_rank={local_rank}: '
            f'{sitecustomize_path}',
            flush=True,
        )


_load_geobench_runtime_sitecustomize()

if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()
