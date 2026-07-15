"""Local runtime patches for GeoBench evaluation."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


TARGET_MODULE = "vllm.model_executor.models.qwen3_next"


def _enabled() -> bool:
    value = os.environ.get("GEOBENCH_FORCE_NATIVE_GDN", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _patch_qwen3_next_gdn(module) -> None:
    if getattr(module, "_GEOBENCH_NATIVE_GDN_PATCHED", False):
        return
    cls = getattr(module, "ChunkGatedDeltaRule", None)
    if cls is None or not hasattr(cls, "forward_native"):
        return

    cls.forward_cuda = cls.forward_native
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._forward_method = self.forward_native

    cls.__init__ = patched_init
    module._GEOBENCH_NATIVE_GDN_PATCHED = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        _patch_qwen3_next_gdn(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)

        if spec is not None and spec.loader is not None:
            spec.loader = _PatchLoader(spec.loader)
        return spec


if _enabled():
    loaded = sys.modules.get(TARGET_MODULE)
    if loaded is not None:
        _patch_qwen3_next_gdn(loaded)
    else:
        sys.meta_path.insert(0, _PatchFinder())
