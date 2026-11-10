"""
Cross-platform device detection and on-demand model manager for Qwen3-TTS.
Supports NVIDIA CUDA (Windows/Linux), Apple Silicon MPS (macOS), and CPU fallback.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger("qwen3_tts.device_utils")


@dataclass
class DeviceConfig:
    device: str
    dtype: torch.dtype
    attn_implementation: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"DeviceConfig(device={self.device}, dtype={self.dtype}, "
            f"attn_implementation={self.attn_implementation})"
        )


def check_flash_attention_available() -> bool:
    """Check if flash-attn is installed and CUDA is available."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def get_optimal_device_config(preferred_device: Optional[str] = None) -> DeviceConfig:
    """
    Determine the optimal device, dtype, and attention implementation.
    
    Priority:
    1. preferred_device if explicitly passed and valid.
    2. CUDA (NVIDIA GPUs on Windows/Linux)
    3. MPS (Apple Silicon GPUs on macOS)
    4. CPU fallback
    """
    dev = (preferred_device or "").strip().lower()

    # Explicit override: CPU
    if dev == "cpu":
        return DeviceConfig(
            device="cpu",
            dtype=torch.float32,
            attn_implementation="sdpa",
        )

    # Explicit override: CUDA or auto-detected CUDA
    if dev.startswith("cuda") or (not dev and torch.cuda.is_available()):
        target_device = dev if dev.startswith("cuda") else "cuda:0"
        # Check bfloat16 capability
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Check FlashAttention 2
        attn_impl = "flash_attention_2" if check_flash_attention_available() else "sdpa"
        return DeviceConfig(
            device=target_device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

    # Explicit override: MPS or auto-detected MPS
    if dev == "mps" or (not dev and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return DeviceConfig(
            device="mps",
            dtype=torch.float32,
            attn_implementation="sdpa",
        )

    # Default fallback: CPU
    return DeviceConfig(
        device="cpu",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )


class ModelManager:
    """
    Manages lazy loading and memory eviction for Qwen3-TTS models.
    Ensures memory-constrained machines (e.g. 16 GB Apple Silicon Macs)
    only keep the active model in memory to prevent OOM.
    """

    def __init__(self, preferred_device: Optional[str] = None):
        self.preferred_device = preferred_device
        self._current_key: Optional[str] = None
        self._current_model: Optional[object] = None
        self._config: DeviceConfig = get_optimal_device_config(preferred_device)

    @property
    def config(self) -> DeviceConfig:
        return self._config

    @staticmethod
    def get_model_id(model_type: str, model_size: str) -> str:
        """Construct Hugging Face repository ID for given type and size."""
        # Standard types: CustomVoice, VoiceDesign, Base
        clean_type = model_type.strip()
        clean_size = model_size.strip()
        return f"Qwen/Qwen3-TTS-12Hz-{clean_size}-{clean_type}"

    def evict_current_model(self) -> None:
        """Unload active model and free GPU/MPS/CPU memory cache."""
        if self._current_model is not None:
            logger.info("Evicting active model: %s", self._current_key)
            del self._current_model
            self._current_model = None
            self._current_key = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception as e:
                    logger.debug("Failed to empty MPS cache: %s", e)

    def get_model(self, model_type: str, model_size: str) -> object:
        """
        Retrieve or load the requested model. If a different model is loaded,
        the previous one will be evicted to preserve system memory.
        """
        model_id = self.get_model_id(model_type, model_size)

        if self._current_key == model_id and self._current_model is not None:
            return self._current_model

        # Evict old model
        self.evict_current_model()

        logger.info("Loading model %s using %s...", model_id, self._config)

        # Import dynamically so device_utils can be tested without downloading models
        from qwen_tts import Qwen3TTSModel
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(model_id)

        try:
            model = Qwen3TTSModel.from_pretrained(
                local_path,
                device_map=self._config.device,
                dtype=self._config.dtype,
                attn_implementation=self._config.attn_implementation,
            )
        except Exception as e:
            # If MPS fails for any backend reason, fallback to CPU
            if self._config.device == "mps":
                logger.warning(
                    "Failed to load model on MPS (%s). Falling back to CPU.", e
                )
                self._config = DeviceConfig(
                    device="cpu", dtype=torch.float32, attn_implementation="sdpa"
                )
                model = Qwen3TTSModel.from_pretrained(
                    local_path,
                    device_map="cpu",
                    dtype=torch.float32,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self._current_key = model_id
        self._current_model = model
        return model
