from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

import app


@pytest.fixture
def history():
    clips = []
    yield clips
    app.cleanup_history(clips)


def test_history_wav_roundtrip_and_newest_first(history):
    audio = (24000, np.linspace(-0.5, 0.5, 24000, dtype=np.float32))
    for mode in app.MODE_CLASSES:
        output, status, updated = app.record_generation((audio, "Done!"), mode, history)
        history[:] = updated
        assert output is audio
        assert 'role="status"' in status
        assert history[0].mode == mode
        samples, rate = sf.read(history[0].path)
        assert rate == 24000
        assert np.allclose(samples, audio[1], atol=1 / 32768)
        assert sf.info(history[0].path).subtype == "PCM_16"
    assert [clip.mode for clip in history] == list(reversed(app.MODE_CLASSES))
    assert len({clip.id for clip in history}) == 3
    assert all(Path(clip.path).suffix == ".wav" for clip in history)


def test_history_is_isolated_and_cleaned_up(history):
    _, _, updated = app.record_generation(
        ((16000, np.zeros(16000)), "Done!"), "Voice Clone", history,
    )
    assert history == []
    history[:] = updated
    path = Path(history[0].path)
    assert path.exists()
    app.cleanup_history(history)
    assert not path.exists()
    app.cleanup_history(history)


def test_failed_generation_does_not_replace_player_or_history(history):
    output, status, updated = app.record_generation(
        (None, "Error: Missing text <script>"), "Voice Design", history,
    )
    assert output == gr.skip()
    assert updated == gr.skip()
    assert 'data-error="true"' in status
    assert "&lt;script&gt;" in status
    assert history == []


def test_history_write_failure_removes_partial_file(history, monkeypatch, tmp_path):
    partial = tmp_path / "partial.wav"

    def fail_write(path, *args, **kwargs):
        partial.write_text(str(path))
        raise OSError("Disk full")

    monkeypatch.setattr(app.sf, "write", fail_write)
    audio = (16000, np.zeros(160))
    with pytest.warns(UserWarning, match="Disk full"):
        output, status, updated = app.record_generation((audio, "Done!"), "Voice Design", history)
    assert output is audio
    assert updated == gr.skip()
    assert "could not be saved" in status
    assert not Path(partial.read_text()).exists()
    assert history == []


def test_reference_metadata_reflects_trim_and_clear():
    audio = (24000, np.zeros((72000, 2), dtype=np.int16))
    assert "3.00 s" in app.reference_audio_info(audio)
    assert "24,000 Hz" in app.reference_audio_info(audio)
    assert "1.00 s" in app.reference_audio_info((24000, audio[1][:24000]))
    assert "Upload or record" in app.reference_audio_info(None)


@pytest.mark.parametrize("audio", [(0, np.zeros(10)), (24000, np.zeros(0))])
def test_reference_metadata_rejects_invalid_audio(audio):
    with pytest.raises(gr.Error):
        app.reference_audio_info(audio)


@pytest.mark.parametrize(
    "callback,generator,args,mode",
    [
        (app.design_with_history, "generate_voice_design", ["Text", "Auto", "Warm"], "Voice Design"),
        (
            app.clone_with_history, "generate_voice_clone",
            [(24000, np.zeros(24000)), "", "Text", "Auto", True, "0.6B"], "Voice Clone",
        ),
        (
            app.custom_with_history, "generate_custom_voice",
            ["Text", "English", "Aiden", "Whisper", "1.7B"], "Custom Voice",
        ),
    ],
)
def test_generation_wrappers_forward_inputs_and_progress(
    callback, generator, args, mode, history, monkeypatch,
):
    progress = object()

    def generate(*actual_args, **kwargs):
        assert actual_args == tuple(args)
        assert kwargs["progress"] is progress
        return (24000, np.zeros(24000)), "Done!"

    monkeypatch.setattr(app, generator, generate)
    _, _, updated = callback(*args, history, progress=progress)
    history[:] = updated
    assert history[0].mode == mode


def test_studio_wiring_and_presets():
    demo = app.build_ui()
    generation_events = [
        event for event in demo.fns.values()
        if event.fn in {app.design_with_history, app.clone_with_history, app.custom_with_history}
    ]
    assert len(generation_events) == 3
    assert all(event.concurrency_id == "tts-generation" for event in generation_events)
    assert all(event.concurrency_limit == 1 for event in generation_events)
    assert all(isinstance(event.inputs[-1], gr.State) for event in generation_events)
    preset_events = [
        event for event in demo.fns.values()
        if event.fn is not None and event.fn.__name__ == "<lambda>"
    ]
    assert {event.fn() for event in preset_events} == {
        *app.DESIGN_PRESETS.values(), *app.STYLE_PRESETS.values(),
    }
    reference = next(
        block for block in demo.blocks.values()
        if isinstance(block, gr.Audio) and block.label == "Reference Audio (Upload or Record)"
    )
    assert reference.editable
    assert reference.interactive
    assert reference.type == "numpy"
