#!/usr/bin/env python3
"""
Qwen3-TTS Audio Studio for Gradio 6+
Supports Voice Design, Voice Clone (Base), and Custom Voice.
Cross-platform compatible with NVIDIA CUDA (Windows/Linux), Apple Silicon MPS (macOS), and CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from html import escape
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr
import numpy as np
import soundfile as sf

from device_utils import ModelManager, is_model_fully_cached

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
    model_id = manager.get_model_id(model_type, model_size)
    cached = is_model_fully_cached(model_id)

    if not cached:
        gr.Info(
            f"⏳ First-time setup: Downloading {model_type} {model_size} weights (~1.5–3.5 GB). "
            "This happens ONLY ONCE! Check terminal for live progress."
        )
        progress(0.05, desc=f"⏳ Downloading {model_id} (~1.5–3.5 GB, first time only)...")
    else:
        progress(0.15, desc=f"🚀 Loading {model_type} {model_size} from local disk cache...")

    try:
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


DESIGN_PRESETS = {
    "Narrator": "A warm, resonant narrator with clear diction and a measured, storytelling pace.",
    "Upbeat Podcast Host": "An upbeat, conversational podcast host with a bright, friendly voice and lively energy.",
    "Calm Meditation Guide": "A calm, soothing meditation guide with a soft voice, gentle pacing, and relaxed delivery.",
    "News Anchor": "A composed news anchor with a clear, balanced voice, precise diction, and confident delivery.",
}

STYLE_PRESETS = {
    "Excited": "Speak with excitement, bright energy, and an enthusiastic, lively rhythm.",
    "Whisper": "Whisper softly and gently, with an intimate, quiet delivery.",
    "Authoritative": "Speak with a confident, authoritative tone, clear emphasis, and a steady pace.",
    "Sad": "Speak softly with a subdued, melancholy tone and a slower, reflective pace.",
}

MODE_CLASSES = {
    "Voice Design": "mode-design",
    "Voice Clone": "mode-clone",
    "Custom Voice": "mode-custom",
}

MODE_WAVEFORM_COLORS = {
    "Voice Design": "#8b5cf6",
    "Voice Clone": "#0e7490",
    "Custom Voice": "#059669",
}

THEME = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    font=["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
).set(
    body_background_fill="#f5f6fa",
    body_background_fill_dark="#0c1019",
    body_text_color="#172033",
    body_text_color_dark="#eef2ff",
    body_text_color_subdued="#526078",
    body_text_color_subdued_dark="#b4bfd2",
    body_text_size="15px",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#171e2b",
    background_fill_secondary="#f0f3f9",
    background_fill_secondary_dark="#101725",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_color="#d5dce7",
    block_border_color_dark="#3b465c",
    block_border_width="0px",
    block_padding="0px",
    block_shadow="none",
    block_radius="12px",
    layout_gap="16px",
    form_gap_width="18px",
    input_radius="12px",
    input_padding="12px 14px",
    input_text_size="15px",
    input_background_fill="#ffffff",
    input_background_fill_dark="#101725",
    input_border_width="1px",
    input_border_color="#d5dce7",
    input_border_color_dark="#3b465c",
    input_border_color_focus="#6d28d9",
    input_border_color_focus_dark="#c4b5fd",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#94a3b8",
    input_shadow="none",
    input_shadow_focus="none",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#172033",
    block_label_text_color_dark="#e2e8f0",
    block_label_border_width="0px",
    block_label_padding="0px",
    block_label_text_size="13px",
    block_label_text_weight="600",
    block_label_shadow="none",
    block_info_text_size="13px",
    prose_text_size="15px",
    button_small_text_size="13px",
    button_small_text_weight="500",
    button_large_text_size="15px",
    button_large_radius="12px",
    button_secondary_background_fill="#f0f3f9",
    button_secondary_background_fill_dark="#202a3a",
    button_secondary_background_fill_hover="#e7ecf5",
    button_secondary_background_fill_hover_dark="#2a3649",
    button_secondary_text_color="#172033",
    button_secondary_text_color_dark="#eef2ff",
    button_secondary_border_color="#d5dce7",
    button_secondary_border_color_dark="#3b465c",
    button_transition="background-color 120ms ease, border-color 120ms ease, color 120ms ease",
)

CUSTOM_CSS = """
.gradio-container {
    --studio-glass: rgba(255, 255, 255, .86);
    --studio-surface: #fff;
    --studio-inset: #f0f3f9;
    --studio-ink: #172033;
    --studio-muted: #526078;
    --studio-line: #d5dce7;
    --studio-violet: #6d28d9;
    --studio-cyan: #0e7490;
    --studio-emerald: #047857;
    --studio-shadow: 0 .5rem 2rem rgba(23, 32, 51, .04);
    width: 100% !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: clamp(1.25rem, 3vw, 2rem) clamp(1rem, 3vw, 2.5rem) !important;
    overflow: visible !important;
    color: var(--studio-ink);
    font-optical-sizing: auto;
    background: transparent;
}
.gradio-container .main {
    max-width: none !important;
    padding: 0 !important;
}
.dark .gradio-container {
    --studio-glass: rgba(23, 30, 43, .9);
    --studio-surface: #171e2b;
    --studio-inset: #101725;
    --studio-ink: #eef2ff;
    --studio-muted: #b4bfd2;
    --studio-line: #3b465c;
    --studio-violet: #c4b5fd;
    --studio-cyan: #67e8f9;
    --studio-emerald: #6ee7b7;
    --studio-shadow: 0 .5rem 2rem rgba(0, 0, 0, .12);
}
.studio-html, .studio-html .html-container {
    padding: 0 !important;
    min-height: 0;
    border: 0 !important;
    background: transparent !important;
}
.studio-header {padding: .25rem 0 .5rem;}
.studio-eyebrow {
    color: var(--studio-muted); font-size: .6875rem;
    font-weight: 650; letter-spacing: .1em; text-transform: uppercase;
}
.studio-header h1 {
    color: var(--studio-ink); font-size: clamp(2rem, 4vw, 2.75rem);
    font-weight: 700; line-height: 1.12; letter-spacing: -.035em; margin: .625rem 0;
}
.studio-header p {color: var(--studio-muted); font-size: .9375rem; line-height: 1.6; margin: 0;}
.hardware-strip {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: .5rem 1.25rem; padding: .875rem 1rem;
    background: var(--studio-glass); border: 1px solid var(--studio-line);
    border-radius: .875rem; color: var(--studio-muted); font-size: .8125rem; line-height: 1.5;
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
}
.hardware-strip > span {white-space: nowrap;}
.hardware-strip strong {color: var(--studio-ink);}
.hardware-strip .runtime-note {margin-left: auto; white-space: normal;}
.studio-help {
    border: 1px solid var(--studio-line) !important;
    border-radius: .875rem !important; background: transparent !important;
}
.studio-help > button {min-height: 2.75rem; padding: .75rem 1rem; font-size: .8125rem;}
.studio-help .prose {color: var(--studio-muted); font-size: .8125rem; line-height: 1.65;}
.studio-help .prose p {margin: 0;}
#studio-tabs {margin-top: .25rem; border: 0; background: transparent;}
#studio-tabs .tab-wrapper {
    min-height: 3.5rem; height: auto; padding: .3125rem;
    margin-bottom: 1.25rem; border: 1px solid var(--studio-line);
    border-radius: 1rem; background: var(--studio-inset);
}
#studio-tabs [role="tablist"] {
    flex: 1; align-items: stretch; gap: .25rem;
    height: auto !important; min-height: 2.75rem; padding: 0;
    border: 0; overflow: visible;
}
/* Gradio measures hidden copies of the tabs; keep their sizing in sync. */
#studio-tabs .tab-container > button {
    display: flex; align-items: center; justify-content: center; text-align: center;
    min-height: 2.75rem; padding: .625rem 1rem;
    border: 1px solid transparent; border-radius: .6875rem;
    font-size: .9375rem; font-weight: 600; line-height: 1.25;
    white-space: nowrap;
}
#studio-tabs .tab-container.visually-hidden {visibility: hidden;}
#studio-tabs [role="tablist"]::after, #studio-tabs [role="tab"]::after {display: none;}
#studio-tabs [role="tab"] {flex: 1 1 0; color: var(--tab-accent);}
#studio-tabs [role="tab"]:hover {background: var(--studio-glass);}
#studio-tabs .tab-container > button:nth-child(1) {--tab-accent: var(--studio-violet);}
#studio-tabs .tab-container > button:nth-child(2) {--tab-accent: var(--studio-cyan);}
#studio-tabs .tab-container > button:nth-child(3) {--tab-accent: var(--studio-emerald);}
#studio-tabs [role="tab"][aria-selected="true"] {
    border-color: var(--studio-line); background: var(--studio-surface);
    box-shadow: 0 1px 3px rgba(0, 0, 0, .06);
}
#studio-tabs [role="tabpanel"] {padding: 0; border: 0; background: transparent;}
.mode-design {
    --mode-accent: var(--studio-violet); --mode-solid: #6d28d9; --mode-hover: #5b21b6;
}
.mode-clone {
    --mode-accent: var(--studio-cyan); --mode-solid: #0e7490; --mode-hover: #155e75;
}
.mode-custom {
    --mode-accent: var(--studio-emerald); --mode-solid: #047857; --mode-hover: #065f46;
}
.mode-design, .mode-clone, .mode-custom {
    --input-border-color-focus: var(--mode-accent);
    --checkbox-background-color-selected: var(--mode-solid);
    --checkbox-border-color-selected: var(--mode-solid);
}
.studio-workspace {
    display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
    align-items: start; gap: 1.25rem;
}
.studio-workspace > .column {min-width: 0 !important;}
.studio-panel {
    min-width: 0 !important; gap: 1.125rem;
    padding: clamp(1rem, 2vw, 1.5rem) !important;
    border: 1px solid var(--studio-line); border-radius: 1.25rem;
    background: var(--studio-glass); box-shadow: var(--studio-shadow);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
}
.studio-panel .form {
    gap: 1.125rem; border: 0; background: transparent; box-shadow: none; overflow: visible;
}
.studio-section-heading .prose h2 {
    color: var(--studio-ink); margin: 0 0 .375rem;
    font-size: 1.1875rem; font-weight: 650; line-height: 1.3; letter-spacing: -.02em;
}
.studio-section-heading .prose p {
    color: var(--studio-muted); margin: 0; font-size: .875rem; line-height: 1.6;
}
.studio-field {
    min-width: 0 !important; padding: 0 !important; overflow: visible !important;
    border: 0 !important; box-shadow: none !important; background: transparent !important;
}
.studio-field textarea {line-height: 1.6; resize: vertical;}
.studio-field textarea::placeholder, .studio-field input::placeholder {opacity: 1;}
.studio-field [data-testid="block-info"] {font-size: .8125rem; font-weight: 600;}
.studio-settings {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .875rem;
}
/* Gradio groups consecutive inputs in a form wrapper; each field belongs in the grid. */
.studio-settings > .form {display: contents;}
.studio-settings-three {grid-template-columns: repeat(3, minmax(0, 1fr));}
.studio-hint .prose, .reference-info {
    color: var(--studio-muted); font-size: .8125rem; line-height: 1.6;
}
.studio-hint .prose p {margin: 0;}
.studio-model-note .prose {font-size: .75rem;}
.gradio-container button, .gradio-container a {touch-action: manipulation;}
.gradio-container :is(button, a, input, textarea):focus-visible {
    outline: 2px solid var(--mode-accent, var(--studio-violet)) !important;
    outline-offset: 3px;
}
.studio-field textarea:focus-visible {
    border-color: var(--mode-accent) !important; outline-offset: 1px;
}
.studio-generate {
    background: var(--mode-solid) !important; border-color: var(--mode-solid) !important;
    color: #fff !important; min-height: 3rem; border-radius: .75rem !important;
    font-size: .9375rem !important; font-weight: 600 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, .08);
    transition: background-color 120ms ease, transform 100ms ease;
}
.studio-generate:hover {background: var(--mode-hover) !important;}
.studio-generate:active {transform: scale(.985);}
.preset-row {gap: .5rem; align-items: center;}
.preset-pill {
    flex: 0 0 auto !important; width: auto !important; max-width: 100%;
    min-height: 2.75rem !important; padding: .625rem .75rem !important;
    border-radius: 999px !important; white-space: nowrap;
    font-size: .8125rem !important; font-weight: 500 !important; line-height: 1.25 !important;
    color: var(--mode-accent) !important; background: var(--studio-inset) !important;
    border: 1px solid var(--studio-line) !important; box-shadow: none !important;
    transition: background-color 120ms ease, border-color 120ms ease, transform 100ms ease;
}
.preset-pill:hover {
    background: color-mix(in srgb, var(--mode-accent) 8%, var(--studio-surface)) !important;
    border-color: var(--mode-accent) !important;
}
.preset-pill:active {transform: scale(.97);}
.studio-output > .studio-empty {
    padding: 1.75rem 1rem !important; text-align: center;
    border: 1px dashed var(--studio-line) !important;
    border-radius: .875rem; background: var(--studio-inset) !important;
}
.studio-empty svg {display: block; margin: 0 auto .75rem; color: var(--mode-accent);}
.studio-empty h3 {margin: 0 0 .375rem; color: var(--studio-ink); font-size: .9375rem; font-weight: 600;}
.studio-empty p {margin: 0; color: var(--studio-muted); font-size: .8125rem; line-height: 1.6;}
.studio-output:has(audio) .studio-empty, .studio-player:not(:has(audio)) {display: none;}
.studio-player, .studio-reference {
    /* Native playback controls otherwise keep light gray text in both themes. */
    --neutral-400: var(--studio-muted);
    min-width: 0 !important; padding: .875rem !important;
    border: 1px solid var(--studio-line) !important;
    border-radius: .875rem !important; background: var(--studio-inset) !important;
}
.studio-reference {--color-accent: var(--mode-solid);}
.studio-reference #waveform > div::part(region-handle) {width: 12px !important;}
.studio-reference #waveform > div::part(region-handle-left),
.studio-reference #waveform > div::part(region-handle-right) {border-color: var(--mode-accent) !important;}
.studio-status-wrap {min-height: 2.75rem;}
.studio-status {
    display: flex; align-items: baseline; gap: .5rem;
    color: var(--studio-muted); font-size: .8125rem; line-height: 1.6; overflow-wrap: anywhere;
}
.studio-status::before {
    content: ""; flex: 0 0 .375rem; width: .375rem; height: .375rem;
    border-radius: 50%; background: var(--mode-accent, var(--studio-muted));
}
.studio-status[data-error="true"] {color: #b91c1c;}
.dark .studio-status[data-error="true"] {color: #fca5a5;}
.studio-status[data-error="true"]::before {background: currentColor;}
#recent-generations {margin-top: .5rem;}
.history-card {
    gap: .875rem; padding: 1rem !important;
    border-radius: 1rem; box-shadow: none;
    background: var(--studio-surface);
    backdrop-filter: none; -webkit-backdrop-filter: none;
}
.history-meta {display: flex; flex-wrap: wrap; gap: .5rem .75rem; align-items: center;}
.mode-badge {
    display: inline-flex; border: 1px solid currentColor; border-radius: 999px;
    color: var(--mode-accent); padding: .1875rem .625rem; font-size: .75rem; font-weight: 600;
    line-height: 1.5; white-space: nowrap;
}
.history-meta time {color: var(--studio-muted); font-size: .75rem; line-height: 1.5;}
.history-playback {
    display: grid; grid-template-columns: minmax(0, 1fr) max-content;
    align-items: center; gap: 1rem;
}
.studio-download {
    flex: 0 0 auto !important;
    min-height: 2.75rem !important; padding: .625rem 1rem !important;
    border-radius: .75rem !important; font-size: .8125rem !important;
    font-weight: 600 !important; white-space: nowrap;
}
.history-empty {
    margin: 0; padding: 1.25rem; border: 1px dashed var(--studio-line);
    border-radius: .875rem; background: var(--studio-inset);
    color: var(--studio-muted); font-size: .875rem; line-height: 1.6; text-align: center;
}
/* Gradio scopes media queries inside its content container, not the page root. */
@media (max-width: 900px) {
    .studio-workspace {grid-template-columns: minmax(0, 1fr);}
}
@media (max-width: 640px) {
    .hardware-strip {gap: .375rem 1rem;}
    .hardware-strip .runtime-note {flex-basis: 100%; margin-left: 0;}
    #studio-tabs .tab-container > button {padding: .625rem .5rem; font-size: .8125rem;}
    .studio-settings-three {grid-template-columns: repeat(2, minmax(0, 1fr));}
    .studio-settings-three .studio-field:last-child {grid-column: 1 / -1;}
    .history-playback {grid-template-columns: minmax(0, 1fr);}
    .history-playback .studio-download {width: 100% !important;}
}
@media (max-width: 380px) {
    #studio-tabs .tab-container > button {font-size: .75rem; padding: .625rem .375rem;}
}
@media (prefers-reduced-motion: reduce) {
    #studio-tabs *, #studio-tabs *::before, #studio-tabs *::after,
    #recent-generations *, #recent-generations *::before, #recent-generations *::after {
        animation: none !important; transition: none !important; scroll-behavior: auto !important;
    }
    .studio-generate:active, .preset-pill:active {transform: none;}
}
@media (prefers-reduced-transparency: reduce) {
    .studio-panel, .hardware-strip {
        background: var(--studio-surface) !important;
        backdrop-filter: none; -webkit-backdrop-filter: none;
    }
}
@media (prefers-contrast: more) {
    .studio-panel, .hardware-strip {
        background: var(--studio-surface) !important;
        border: 2px solid var(--studio-ink) !important;
        backdrop-filter: none; -webkit-backdrop-filter: none;
    }
    .studio-hint .prose, .studio-status, .reference-info, .history-meta time {color: var(--studio-ink);}
}
"""

TAB_KEYBOARD_JS = """
() => {
    const navigation = document.querySelector("#studio-tabs");
    if (!navigation) throw new Error("Studio mode navigation is unavailable.");
    if (navigation.dataset.keyboardReady) return [];
    navigation.dataset.keyboardReady = "true";
    const tabs = () => [...navigation.querySelectorAll('[role="tab"]')]
        .filter(tab => tab.getAttribute("aria-disabled") !== "true");
    const setTabStop = active => tabs().forEach(tab => { tab.tabIndex = tab === active ? 0 : -1; });
    setTabStop(navigation.querySelector('[role="tab"][aria-selected="true"]'));
    navigation.addEventListener("click", event => {
        const tab = event.target.closest('[role="tab"]');
        if (tab) setTabStop(tab);
    });
    navigation.addEventListener("keydown", event => {
        const current = event.target.closest('[role="tab"]');
        if (!current) return;
        const items = tabs();
        const index = items.indexOf(current);
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % items.length;
        else if (event.key === "ArrowLeft") next = (index - 1 + items.length) % items.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = items.length - 1;
        else return;
        event.preventDefault();
        items[next].focus();
        items[next].click();
    });
    return [];
}
"""


@dataclass(frozen=True)
class GenerationClip:
    id: str
    timestamp: str
    mode: str
    path: str


def cleanup_history(history: List[GenerationClip]) -> None:
    """Release session-owned WAVs when Gradio expires the browser session."""
    for clip in history:
        try:
            Path(clip.path).unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove session WAV %s", clip.path)


def status_markup(message: str) -> str:
    is_error = str(message.startswith("Error:")).lower()
    return (
        f'<div class="studio-status" role="status" aria-live="polite" '
        f'data-error="{is_error}">{escape(message)}</div>'
    )


def reference_audio_info(audio: Any) -> str:
    """Describe the current reference, including edits made in the waveform editor."""
    try:
        converted = audio_to_tuple(audio)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Cannot read reference audio: {exc}") from exc
    if converted is None:
        message = "Upload or record a reference to see its duration and sample rate."
    else:
        wav, sample_rate = converted
        if sample_rate <= 0 or wav.size == 0:
            raise gr.Error("Reference audio must contain samples and have a positive sample rate.")
        message = f"Duration: {len(wav) / sample_rate:.2f} s | Sample rate: {sample_rate:,} Hz"
    return f'<div class="reference-info" role="status" aria-live="polite">{escape(message)}</div>'


def record_generation(
    result: Tuple[Optional[Tuple[int, np.ndarray]], str],
    mode: str,
    history: List[GenerationClip],
) -> Tuple[Any, str, Any]:
    """Keep only file references in session state, never generated audio arrays."""
    audio, message = result
    if audio is None:
        return gr.skip(), status_markup(message), gr.skip()

    clip_id = uuid4().hex
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    filename = f"qwen-{mode.lower().replace(' ', '-')}-{clip_id}-"
    path = None
    try:
        # Close before writing so the file is also accessible on Windows.
        with tempfile.NamedTemporaryFile(prefix=filename, suffix=".wav", delete=False) as file:
            path = file.name
        sample_rate, wav = audio
        sf.write(path, normalize_audio(wav), sample_rate, subtype="PCM_16")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Could not save generated WAV to session history")
        if path is not None:
            cleanup_history([GenerationClip(clip_id, timestamp, mode, path)])
        warning = f"Speech generated, but the history WAV could not be saved: {exc}"
        gr.Warning(warning)
        return audio, status_markup(warning), gr.skip()

    clip = GenerationClip(clip_id, timestamp, mode, path)
    return audio, status_markup(message), [clip, *history]


def design_with_history(
    text: str, language: str, description: str, history: List[GenerationClip],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, Any]:
    return record_generation(
        generate_voice_design(text, language, description, progress=progress),
        "Voice Design", history,
    )


def clone_with_history(
    ref_audio: Any, ref_text: str, target_text: str, language: str,
    xvector: bool, model_size: str, history: List[GenerationClip],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, Any]:
    return record_generation(
        generate_voice_clone(
            ref_audio, ref_text, target_text, language, xvector, model_size, progress=progress,
        ),
        "Voice Clone", history,
    )


def custom_with_history(
    text: str, language: str, speaker: str, instruct: str, model_size: str,
    history: List[GenerationClip], progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[Any, str, Any]:
    return record_generation(
        generate_custom_voice(text, language, speaker, instruct, model_size, progress=progress),
        "Custom Voice", history,
    )


def preset_buttons(presets: Dict[str, str], target: gr.Textbox) -> None:
    with gr.Row(elem_classes="preset-row"):
        for label, prompt in presets.items():
            button = gr.Button(
                label, size="sm", scale=0, min_width=0, elem_classes="preset-pill",
            )
            button.click(
                fn=lambda value=prompt: value,
                inputs=None, outputs=target, queue=False, api_visibility="private",
                show_progress="hidden",
            )


def output_panel(mode: str, color: str) -> Tuple[gr.Audio, gr.HTML]:
    with gr.Column(
        scale=2, min_width=0, elem_id=f"{mode}-preview",
        elem_classes=["studio-panel", "studio-output"],
    ):
        gr.Markdown(
            "## Listen back\nPreview your take before downloading.",
            elem_classes="studio-section-heading",
        )
        gr.HTML(
            '<svg width="48" height="32" viewBox="0 0 48 32" fill="none" aria-hidden="true">'
            '<path d="M4 13v6m8-11v16m8-21v26m8-22v18m8-15v12m8-8v4" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round"/></svg>'
            '<h3>Ready for your first take</h3>'
            '<p>Generate speech to see the waveform<br>and hear your voice in action.</p>',
            elem_classes=["studio-html", "studio-empty"],
        )
        audio = gr.Audio(
            label="Generated speech", type="numpy", format="wav", interactive=False,
            editable=False, buttons=[], min_width=0,
            elem_id=f"{mode}-audio", elem_classes="studio-player",
            waveform_options=gr.WaveformOptions(
                waveform_color="#64748b", waveform_progress_color=color,
            ),
        )
        status = gr.HTML(
            status_markup("Ready when you are."),
            elem_id=f"{mode}-status", elem_classes=["studio-html", "studio-status-wrap"],
        )
    return audio, status


def build_ui(preferred_device: Optional[str] = None) -> gr.Blocks:
    """Build the studio without eagerly loading any model weights."""
    manager = get_model_manager(preferred_device)

    with gr.Blocks(title="Qwen3-TTS Audio Studio") as demo:
        history = gr.State([], delete_callback=cleanup_history)
        gr.HTML(
            '<header class="studio-header"><div class="studio-eyebrow">Qwen3-TTS / Audio Studio</div>'
            '<h1>Give your words a voice.</h1>'
            '<p>Design a new voice, work from a reference, or direct a studio speaker.</p></header>',
            elem_id="studio-header", elem_classes="studio-html",
        )

        def hardware_markup() -> str:
            active = manager.config
            precision = str(active.dtype).replace("torch.", "")
            return (
                '<div class="hardware-strip" role="status" aria-live="polite">'
                f'<span>Device <strong>{escape(active.device.upper())}</strong></span>'
                f'<span>Precision <strong>{escape(precision)}</strong></span>'
                f'<span>Attention <strong>{escape(str(active.attn_implementation))}</strong></span>'
                '<span class="runtime-note">Models load on demand / one model in memory</span></div>'
            )

        device_info = gr.HTML(
            hardware_markup(), elem_id="studio-runtime", elem_classes="studio-html",
        )
        with gr.Accordion("Model downloads & memory", open=False, elem_classes="studio-help"):
            gr.Markdown(
                "The first generation for each model downloads its weights "
                "(approximately 1.5-3.5 GB). Later requests reuse the local disk cache. "
                "Switching models unloads the previous model to conserve memory. "
                "Unsupported Apple Silicon operations automatically retry on CPU."
            )

        with gr.Tabs(elem_id="studio-tabs"):
            with gr.Tab(
                "Voice Design", id="design", elem_id="voice-design", elem_classes="mode-design",
            ):
                with gr.Row(elem_classes="studio-workspace"):
                    with gr.Column(
                        scale=3, min_width=0, elem_id="design-controls", elem_classes="studio-panel",
                    ):
                        gr.Markdown(
                            "## Imagine a voice\nDescribe the character, tone, and delivery.",
                            elem_classes="studio-section-heading",
                        )
                        design_text = gr.Textbox(
                            label="Script",
                            placeholder="Enter the text you want the model to speak...",
                            lines=4,
                            value="Welcome to Qwen3-TTS! This voice is generated entirely from a text description.",
                            elem_id="design-text", elem_classes="studio-field",
                        )
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            elem_classes="studio-field",
                        )
                        design_instruct = gr.Textbox(
                            label="Voice description",
                            placeholder="e.g. A young cheerful female voice speaking with enthusiasm and warmth.",
                            lines=3,
                            value="A warm and clear female voice speaking with a friendly tone.",
                            elem_id="design-instruction", elem_classes="studio-field",
                        )
                        gr.Markdown("Start with an inspiration", elem_classes="studio-hint")
                        preset_buttons(DESIGN_PRESETS, design_instruct)
                        gr.Markdown(
                            "Uses the 1.7B Voice Design model.",
                            elem_classes=["studio-hint", "studio-model-note"],
                        )
                        design_btn = gr.Button(
                            "Generate voice", variant="primary",
                            elem_id="design-generate", elem_classes="studio-generate",
                        )
                    design_audio, design_status = output_panel(
                        "design", MODE_WAVEFORM_COLORS["Voice Design"],
                    )

            with gr.Tab(
                "Voice Clone", id="clone", elem_id="voice-clone", elem_classes="mode-clone",
            ):
                with gr.Row(elem_classes="studio-workspace"):
                    with gr.Column(
                        scale=3, min_width=0, elem_id="clone-controls", elem_classes="studio-panel",
                    ):
                        gr.Markdown(
                            "## Start with a voice\nUse a clean reference of at least 3 seconds.",
                            elem_classes="studio-section-heading",
                        )
                        gr.Markdown(
                            "Only clone a voice you own or have permission to use.",
                            elem_classes="studio-hint",
                        )
                        clone_target_text = gr.Textbox(
                            label="Script",
                            placeholder="Enter the text for the cloned voice to speak...",
                            lines=3,
                            value="Hello, this audio was synthesized using voice cloning from a reference sample.",
                            elem_id="clone-text", elem_classes="studio-field",
                        )
                        clone_ref_audio = gr.Audio(
                            label="Reference audio",
                            sources=["upload", "microphone"],
                            type="numpy",
                            interactive=True,
                            editable=True,
                            buttons=[],
                            min_width=0, elem_id="clone-reference", elem_classes="studio-reference",
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#64748b",
                                waveform_progress_color=MODE_WAVEFORM_COLORS["Voice Clone"],
                                trim_region_color="rgba(6, 182, 212, 0.2)",
                                show_recording_waveform=True,
                                skip_length=5,
                            ),
                        )
                        clone_ref_info = gr.HTML(
                            reference_audio_info(None),
                            elem_id="clone-reference-info", elem_classes="studio-html",
                        )
                        gr.Markdown(
                            "Play or seek to review. Choose the scissors tool, drag the selection "
                            "handles, then select Trim. Update the transcript to match the trimmed audio.",
                            elem_classes="studio-hint",
                        )
                        clone_ref_audio.change(
                            reference_audio_info, inputs=clone_ref_audio, outputs=clone_ref_info,
                            queue=False, api_visibility="private", show_progress="hidden",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference transcript",
                            info="Optional when x-vector only is enabled.",
                            placeholder="Exact transcript of the reference audio clip (improves clone fidelity)...",
                            lines=2,
                            elem_id="clone-transcript", elem_classes="studio-field",
                        )
                        with gr.Row(elem_classes="studio-settings"):
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                min_width=0, elem_classes="studio-field",
                            )
                            clone_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                min_width=0, elem_classes="studio-field",
                            )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only",
                            info="A faster, text-free clone without a reference transcript.",
                            value=False,
                            elem_id="clone-xvector", elem_classes="studio-field",
                        )
                        clone_btn = gr.Button(
                            "Clone voice", variant="primary",
                            elem_id="clone-generate", elem_classes="studio-generate",
                        )
                    clone_audio, clone_status = output_panel(
                        "clone", MODE_WAVEFORM_COLORS["Voice Clone"],
                    )

            with gr.Tab(
                "Custom Voice", id="custom", elem_id="custom-voice", elem_classes="mode-custom",
            ):
                with gr.Row(elem_classes="studio-workspace"):
                    with gr.Column(
                        scale=3, min_width=0, elem_id="custom-controls", elem_classes="studio-panel",
                    ):
                        gr.Markdown(
                            "## Direct the delivery\nNine speakers. Your words. Your direction.",
                            elem_classes="studio-section-heading",
                        )
                        custom_text = gr.Textbox(
                            label="Script",
                            placeholder="Enter text to synthesize...",
                            lines=4,
                            value="Hello everyone! Qwen3-TTS offers expressive and natural sounding speech generation.",
                            elem_id="custom-text", elem_classes="studio-field",
                        )
                        with gr.Row(elem_classes=["studio-settings", "studio-settings-three"]):
                            custom_speaker = gr.Dropdown(
                                label="Preset Speaker",
                                choices=SPEAKERS,
                                value="Aiden",
                                min_width=0, elem_classes="studio-field",
                            )
                            custom_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                min_width=0, elem_classes="studio-field",
                            )
                            custom_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                min_width=0, elem_classes="studio-field",
                            )
                        custom_instruct = gr.Textbox(
                            label="Style / emotion",
                            placeholder="e.g. Speak excitedly and fast, or whisper gently...",
                            lines=2,
                            elem_id="custom-instruction", elem_classes="studio-field",
                        )
                        preset_buttons(STYLE_PRESETS, custom_instruct)
                        gr.Markdown(
                            "Natural-language style control is supported by the 1.7B model. "
                            "The 0.6B model may not follow these instructions.",
                            elem_classes="studio-hint",
                        )
                        custom_btn = gr.Button(
                            "Synthesize speech", variant="primary",
                            elem_id="custom-generate", elem_classes="studio-generate",
                        )
                    custom_audio, custom_status = output_panel(
                        "custom", MODE_WAVEFORM_COLORS["Custom Voice"],
                    )

        with gr.Column(elem_id="recent-generations", elem_classes="studio-panel"):
            gr.Markdown(
                "## Recent Generations\n"
                "Your takes from all three modes, newest first. Download clips to keep them after this session.",
                elem_classes="studio-section-heading",
            )

            @gr.render(inputs=history)
            def render_history(clips: List[GenerationClip]):
                if not clips:
                    gr.HTML(
                        '<p class="history-empty">Your studio is quiet. Generate a clip to start your collection.</p>',
                        elem_classes="studio-html",
                    )
                for clip in clips:
                    mode_class = MODE_CLASSES[clip.mode]
                    timestamp = datetime.fromisoformat(clip.timestamp)
                    with gr.Column(
                        elem_classes=["studio-panel", "history-card", mode_class], key=clip.id,
                    ):
                        gr.HTML(
                            '<div class="history-meta">'
                            f'<span class="mode-badge">{escape(clip.mode)}</span>'
                            f'<time datetime="{escape(clip.timestamp)}">'
                            f'{escape(timestamp.strftime("%b %d, %Y / %H:%M:%S %Z"))}</time></div>',
                            elem_classes="studio-html",
                        )
                        with gr.Row(elem_classes="history-playback"):
                            gr.Audio(
                                value=clip.path,
                                label=f"{clip.mode} / {timestamp.strftime('%H:%M:%S')}",
                                interactive=False, editable=False, format="wav",
                                min_width=0, scale=1, buttons=[], elem_classes="studio-player",
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#64748b",
                                    waveform_progress_color=MODE_WAVEFORM_COLORS[clip.mode],
                                ),
                                key=f"audio-{clip.id}",
                            )
                            gr.DownloadButton(
                                "Download WAV", value=clip.path, min_width=0, scale=0,
                                size="sm", elem_classes="studio-download",
                                key=f"download-{clip.id}",
                            )

        generation_bindings = [
            (
                design_btn, design_with_history,
                [design_text, design_language, design_instruct, history],
                [design_audio, design_status, history],
            ),
            (
                clone_btn, clone_with_history,
                [
                    clone_ref_audio, clone_ref_text, clone_target_text, clone_language,
                    clone_xvector, clone_model_size, history,
                ],
                [clone_audio, clone_status, history],
            ),
            (
                custom_btn, custom_with_history,
                [custom_text, custom_language, custom_speaker, custom_instruct, custom_model_size, history],
                [custom_audio, custom_status, history],
            ),
        ]
        for button, callback, inputs, outputs in generation_bindings:
            event = button.click(
                fn=callback, inputs=inputs, outputs=outputs,
                concurrency_id="tts-generation", concurrency_limit=1,
                show_progress_on=outputs[1],
            )
            event.then(
                hardware_markup, outputs=device_info, queue=False, api_visibility="private",
            )
        demo.load(
            hardware_markup, outputs=device_info, queue=False, api_visibility="private",
            js=TAB_KEYBOARD_JS,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS Audio Studio")
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
