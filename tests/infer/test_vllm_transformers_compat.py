from swift.infer_engine.utils import patch_vllm_transformers_compat


def test_patch_vllm_transformers_compat_accepts_sequence_ignore_keys():
    from transformers.configuration_utils import PretrainedConfig

    class DummyConfig(PretrainedConfig):

        def __init__(self, **kwargs):
            self.rope_parameters = {}
            self.called_with = None
            super().__init__(**kwargs)

        def validate_rope(self, ignore_keys=None):
            self.called_with = ignore_keys

    patch_vllm_transformers_compat()
    config = DummyConfig(ignore_keys_at_rope_validation=['a', 'b'])
    assert config.called_with == {'a', 'b'}


def test_patch_vllm_transformers_compat_patches_vllm_hash_and_engine_core():
    import pytest
    from transformers.configuration_utils import PretrainedConfig

    pytest.importorskip('vllm')
    from vllm.config import utils as vllm_config_utils
    from vllm.v1.engine.core import EngineCoreProc

    patch_vllm_transformers_compat()

    assert isinstance(vllm_config_utils.normalize_value(PretrainedConfig()), str)
    assert getattr(EngineCoreProc.run_engine_core, '_swift_applies_transformers_compat', False)


def test_patch_vllm_transformers_compat_can_force_native_gdn(monkeypatch):
    import pytest

    pytest.importorskip('vllm')
    monkeypatch.setenv('SWIFT_VLLM_FORCE_NATIVE_GDN', '1')

    patch_vllm_transformers_compat()

    from vllm.model_executor.models.qwen3_next import ChunkGatedDeltaRule

    assert getattr(ChunkGatedDeltaRule.__init__, '_swift_force_native_gdn', False)
