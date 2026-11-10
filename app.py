#!/usr/bin/env python3
"""
Qwen3-TTS Unified Gradio Web UI
Supports Voice Design, Voice Clone (Base), and Custom Voice.
Cross-platform compatible with NVIDIA CUDA (Windows/Linux), Apple Silicon MPS (macOS), and CPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from device_utils import ModelManager, get_optimal_device_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3_tts.app")

# Available choices matching official Hugging Face Space
MODEL_SIZES = ["0.6B", "1.7B"]

SPEAKERS = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_fu",
    "Vivian",
]

LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
]

# Global manager initialized in main or lazily
_model_manager: Optional[ModelManager] = None


def get_model_manager(preferred_device: Optional[str] = None) -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(preferred_device=preferred_device)
    return _model_manager


def normalize_audio(wav: Any, eps: float = 1e-12, clip: bool = True) -> np.ndarray:
    """Normalize audio data to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported audio dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """Convert Gradio audio input into a normalized (wav_float32, sample_rate) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2:
        # Support both (sr, wav) and (wav, sr), with int or numpy integer types
        if isinstance(audio[0], (int, np.integer)):
            sr = int(audio[0])
            wav = normalize_audio(audio[1])
            return wav, sr
        elif isinstance(audio[1], (int, np.integer)):
            sr = int(audio[1])
            wav = normalize_audio(audio[0])
            return wav, sr

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = normalize_audio(audio["data"])
        return wav, sr

    return None


def _execute_with_mps_fallback(
    action_fn: Any,
    model_type: str,
    model_size: str,
    progress: gr.Progress,
) -> Any:
    """Execute model action with automatic fallback from MPS to CPU if kernel errors occur."""
    manager = get_model_manager()
    try:
        progress(0.2, desc=f"Loading {model_type} {model_size} model...")
        model = manager.get_model(model_type, model_size)
        progress(0.5, desc="Synthesizing speech...")
        return action_fn(model)
    except Exception as e:
        if manager.config.device == "mps":
            logger.warning("Generation encountered error on MPS (%s). Retrying on CPU...", e)
            manager.switch_to_cpu()
            progress(0.3, desc=f"Reloading {model_type} {model_size} on CPU...")
            model = manager.get_model(model_type, model_size)
            progress(0.6, desc="Synthesizing speech on CPU...")
            return action_fn(model)
        raise


def generate_voice_design(
    text: str,
    language: str,
    voice_description: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate speech using Voice Design model (1.7B)."""
    if not text or not text.strip():
        return None, "Error: Text to synthesize is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    def _run(model):
        return model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

    try:
        wavs, sr = _execute_with_mps_fallback(_run, "VoiceDesign", "1.7B", progress)
        progress(1.0, desc="Done!")
        return (sr, wavs[0]), "Voice design generation completed successfully!"
    except Exception as e:
        logger.exception("Voice design generation failed")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(
    ref_audio: Any,
    ref_text: str,
    target_text: str,
    language: str,
    use_xvector_only: bool,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is disabled."

    def _run(model):
        return model.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )

    try:
        wavs, sr = _execute_with_mps_fallback(_run, "Base", model_size, progress)
        progress(1.0, desc="Done!")
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        logger.exception("Voice clone generation failed")
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    speaker_id = speaker.lower().replace(" ", "_")

    def _run(model):
        return model.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker_id,
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

    try:
        wavs, sr = _execute_with_mps_fallback(_run, "CustomVoice", model_size, progress)
        progress(1.0, desc="Done!")
        return (sr, wavs[0]), "Speech generation completed successfully!"
    except Exception as e:
        logger.exception("Custom voice generation failed")
        return None, f"Error: {type(e).__name__}: {e}"


THEME = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
)

CUSTOM_CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto;}
.tab-content {padding: 16px;}
.status-box {font-weight: 500; font-size: 0.95rem;}
"""


def build_ui(preferred_device: Optional[str] = None) -> gr.Blocks:
    """Build the unified Gradio Blocks UI."""
    manager = get_model_manager(preferred_device)
    config = manager.config

    with gr.Blocks(title="Qwen3-TTS Unified Demo") as demo:
        gr.Markdown(
            f"""
# 🎙️ Qwen3-TTS Unified Web Demo
**Cross-Platform Speech Generation (Apple Silicon & NVIDIA CUDA)**

Active Device Configuration: `{config.device}` | Precision: `{config.dtype}` | Attention: `{config.attn_implementation}`

Three generation modes:
- **Voice Design**: Create custom voice timbres using natural language descriptions.
- **Voice Clone (Base)**: Clone any voice using a 3-second reference audio.
- **TTS (CustomVoice)**: Synthesize speech from 9 preset speakers with natural language style instructions.
"""
        )

        with gr.Tabs():
            # ========================================================
            # Tab 1: Voice Design
            # ========================================================
            with gr.Tab("🎨 Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language Description (1.7B)")
                with gr.Row():
                    with gr.Column(scale=3):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want the model to speak...",
                            lines=4,
                            value="Welcome to Qwen3-TTS! This voice is generated entirely from a text description.",
                        )
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description / Style Instruction",
                            placeholder="e.g. A young cheerful female voice speaking with enthusiasm and warmth.",
                            lines=3,
                            value="A warm and clear female voice speaking with a friendly tone.",
                        )
                        design_btn = gr.Button("✨ Generate Voice", variant="primary")

                    with gr.Column(scale=2):
                        design_audio = gr.Audio(label="Synthesized Speech", type="numpy")
                        design_status = gr.Textbox(label="Status", interactive=False)

                design_btn.click(
                    fn=generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio, design_status],
                )

            # ========================================================
            # Tab 2: Voice Clone (Base)
            # ========================================================
            with gr.Tab("🧬 Voice Clone (Base)"):
                gr.Markdown("### Clone Any Voice from a Reference Audio (3+ Seconds)")
                with gr.Row():
                    with gr.Column(scale=3):
                        clone_target_text = gr.Textbox(
                            label="Target Text to Synthesize",
                            placeholder="Enter the text for the cloned voice to speak...",
                            lines=3,
                            value="Hello, this audio was synthesized using voice cloning from a reference sample.",
                        )
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload or Record)",
                            sources=["upload", "microphone"],
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Audio Transcript (Optional if x-vector only is checked)",
                            placeholder="Exact transcript of the reference audio clip (improves clone fidelity)...",
                            lines=2,
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                            )
                            clone_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                            )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (Fast clone, doesn't require reference text)",
                            value=False,
                        )
                        clone_btn = gr.Button("🧬 Clone Voice", variant="primary")

                    with gr.Column(scale=2):
                        clone_audio = gr.Audio(label="Synthesized Speech", type="numpy")
                        clone_status = gr.Textbox(label="Status", interactive=False)

                clone_btn.click(
                    fn=generate_voice_clone,
                    inputs=[
                        clone_ref_audio,
                        clone_ref_text,
                        clone_target_text,
                        clone_language,
                        clone_xvector,
                        clone_model_size,
                    ],
                    outputs=[clone_audio, clone_status],
                )

            # ========================================================
            # Tab 3: Custom Voice (TTS)
            # ========================================================
            with gr.Tab("🗣️ Custom Voice (TTS)"):
                gr.Markdown("### Synthesize Speech with Preset Speakers and Style Control")
                with gr.Row():
                    with gr.Column(scale=3):
                        custom_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter text to synthesize...",
                            lines=4,
                            value="Hello everyone! Qwen3-TTS offers expressive and natural sounding speech generation.",
                        )
                        with gr.Row():
                            custom_speaker = gr.Dropdown(
                                label="Preset Speaker",
                                choices=SPEAKERS,
                                value="Aiden",
                            )
                            custom_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                            )
                            custom_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                            )
                        custom_instruct = gr.Textbox(
                            label="Style / Emotion Instruction (Optional)",
                            placeholder="e.g. Speak excitedly and fast, or whisper gently...",
                            lines=2,
                        )
                        custom_btn = gr.Button("🔊 Synthesize Speech", variant="primary")

                    with gr.Column(scale=2):
                        custom_audio = gr.Audio(label="Synthesized Speech", type="numpy")
                        custom_status = gr.Textbox(label="Status", interactive=False)

                custom_btn.click(
                    fn=generate_custom_voice,
                    inputs=[
                        custom_text,
                        custom_language,
                        custom_speaker,
                        custom_instruct,
                        custom_model_size,
                    ],
                    outputs=[custom_audio, custom_status],
                )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Unified Gradio Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--device", default=None, help="Device override: cuda, cuda:0, mps, cpu")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link")
    parser.add_argument("--ssl-certfile", default=None, help="SSL cert file for HTTPS")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file for HTTPS")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Qwen3-TTS Web UI...")
    demo = build_ui(preferred_device=args.device)

    launch_kwargs: Dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "theme": THEME,
        "css": CUSTOM_CSS,
    }
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    logger.info("Launching server on %s:%d...", args.host, args.port)
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
