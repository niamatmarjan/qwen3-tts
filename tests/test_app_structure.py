import numpy as np
import pytest
from app import normalize_audio, audio_to_tuple, build_ui, SPEAKERS, LANGUAGES, MODEL_SIZES


def test_normalize_audio_int16():
    int_audio = np.array([-32768, 0, 32767], dtype=np.int16)
    norm = normalize_audio(int_audio)
    assert norm.dtype == np.float32
    assert -1.0 <= norm.min() and norm.max() <= 1.0


def test_normalize_audio_float():
    float_audio = np.array([-0.5, 0.0, 0.8], dtype=np.float32)
    norm = normalize_audio(float_audio)
    assert norm.dtype == np.float32
    assert np.allclose(norm, float_audio)


def test_audio_to_tuple_numpy_integer_sample_rate():
    wav = np.zeros(16000, dtype=np.int16)
    sr_int64 = np.int64(16000)
    res = audio_to_tuple((sr_int64, wav))
    assert res is not None
    audio_data, sample_rate = res
    assert sample_rate == 16000
    assert audio_data.dtype == np.float32

    # Also test np.int32
    sr_int32 = np.int32(24000)
    res32 = audio_to_tuple((sr_int32, wav))
    assert res32 is not None
    _, sample_rate32 = res32
    assert sample_rate32 == 24000


def test_audio_to_tuple_tuple_format():
    wav = np.zeros(16000, dtype=np.int16)
    sr = 16000
    res = audio_to_tuple((sr, wav))
    assert res is not None
    audio_data, sample_rate = res
    assert sample_rate == 16000
    assert audio_data.dtype == np.float32


def test_audio_to_tuple_none():
    assert audio_to_tuple(None) is None


def test_app_constants():
    assert "Aiden" in SPEAKERS
    assert "English" in LANGUAGES
    assert "1.7B" in MODEL_SIZES


def test_ui_builds_without_error():
    demo = build_ui()
    assert demo is not None
