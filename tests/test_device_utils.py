import pytest
import torch
from device_utils import get_optimal_device_config, ModelManager, DeviceConfig


def test_device_config_auto():
    config = get_optimal_device_config()
    assert isinstance(config, DeviceConfig)
    assert config.device in ("cuda", "mps", "cpu")
    assert isinstance(config.dtype, torch.dtype)
    if config.device == "cuda":
        assert config.dtype in (torch.bfloat16, torch.float16)
    elif config.device == "mps":
        assert config.attn_implementation == "sdpa"


def test_device_config_explicit_override():
    config = get_optimal_device_config(preferred_device="cpu")
    assert config.device == "cpu"
    assert config.dtype == torch.float32
    assert config.attn_implementation == "sdpa"


def test_model_manager_key_generation():
    manager = ModelManager()
    key = manager.get_model_id("CustomVoice", "1.7B")
    assert key == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    key2 = manager.get_model_id("VoiceDesign", "1.7B")
    assert key2 == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    key3 = manager.get_model_id("Base", "0.6B")
    assert key3 == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def test_model_manager_eviction():
    manager = ModelManager()
    # Mocking loaded model state
    manager._current_key = "test_key"
    manager._current_model = "fake_model_object"

    manager.evict_current_model()
    assert manager._current_key is None
    assert manager._current_model is None
