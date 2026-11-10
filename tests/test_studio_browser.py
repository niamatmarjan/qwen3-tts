"""Optional real-browser regressions. Set AGENT_BROWSER_EXECUTABLE or install agent-browser."""

import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf

import app


BROWSER_EXECUTABLE = os.environ.get("AGENT_BROWSER_EXECUTABLE") or shutil.which("agent-browser")
pytestmark = pytest.mark.skipif(
    not BROWSER_EXECUTABLE, reason="agent-browser is required for browser regressions",
)

MODES = {
    "design": "voice-design",
    "clone": "voice-clone",
    "custom": "custom-voice",
}


def _serve(port):
    """Replace inference only; exercise the real validation, events, state, and audio UI."""
    cache = Path(os.environ["STUDIO_TEST_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    import tempfile

    tempfile.tempdir = str(cache)
    samples = (0.2 * np.sin(2 * np.pi * 220 * np.arange(30000) / 24000)).astype(np.float32)

    def generate(action_fn, model_type, model_size, progress):
        progress(0.5, desc="Preparing browser-test audio...")
        return [samples], 24000

    app._execute_with_mps_fallback = generate
    app.build_ui("cpu").queue().launch(
        server_name="127.0.0.1", server_port=port, theme=app.THEME, css=app.CUSTOM_CSS,
    )


@pytest.fixture(scope="module")
def studio_url(tmp_path_factory):
    directory = tmp_path_factory.mktemp("studio-browser")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "GRADIO_ANALYTICS_ENABLED": "False",
        "GRADIO_TEMP_DIR": str(directory / "gradio"),
        "STUDIO_TEST_CACHE": str(directory / "wav"),
    }
    url = f"http://127.0.0.1:{port}"
    with (directory / "server.log").open("w+") as log:
        print(f"Studio browser server log: {directory / 'server.log'}")
        process = subprocess.Popen(
            [
                sys.executable, "-c",
                f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_serve']({port})",
            ],
            cwd=Path(__file__).resolve().parents[1], env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(f"{url}/config", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                log.seek(0)
                pytest.fail(f"Studio test server did not start:\n{log.read()}")
            yield url
        finally:
            process.terminate()
            process.wait(timeout=10)


class Browser:
    def __init__(self, session, url):
        self.session = session
        self.url = url

    def __call__(self, *args, stdin=None):
        result = subprocess.run(
            [BROWSER_EXECUTABLE, "--session", self.session, "--json", *args],
            input=stdin, text=True, capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            diagnostic = subprocess.run(
                [
                    BROWSER_EXECUTABLE, "--session", self.session, "--json", "eval",
                    "({url:location.href,ready:document.readyState,text:document.body.innerText.slice(0,2000),"
                    "resources:performance.getEntriesByType('resource').slice(-10).map(e=>({name:e.name,duration:e.duration}))})",
                ],
                text=True, capture_output=True, timeout=15,
            )
            errors = subprocess.run(
                [BROWSER_EXECUTABLE, "--session", self.session, "--json", "errors"],
                text=True, capture_output=True, timeout=15,
            )
            network = subprocess.run(
                [BROWSER_EXECUTABLE, "--session", self.session, "--json", "network", "requests", "--filter", "config"],
                text=True, capture_output=True, timeout=15,
            )
            pytest.fail(
                f"{args}\n{result.stdout}\n{result.stderr}\nPage: {diagnostic.stdout}\n"
                f"Errors: {errors.stdout}\nConfig requests: {network.stdout}"
            )
        payload = json.loads(result.stdout)
        assert payload["success"], payload
        return payload["data"]

    def evaluate(self, script):
        return self("eval", "--stdin", stdin=script)["result"]

    def open(self, theme="dark", width=1440):
        self("set", "viewport", str(width), "1000")
        self("open", f"{self.url}/?__theme={theme}")
        self("wait", "#voice-design-button")
        self.click("#voice-design-button")
        self("wait", "--fn", "document.querySelector('#design-generate') !== null")

    def click(self, selector):
        self("scrollintoview", selector)
        self("click", selector)


@pytest.fixture
def browser(studio_url):
    result = subprocess.run(
        [BROWSER_EXECUTABLE, "session", "id", "--scope", "worktree", "--prefix", "studio-regression"],
        text=True, capture_output=True, check=True,
    )
    instance = Browser(result.stdout.strip(), studio_url)
    try:
        yield instance
    finally:
        instance("close")


WCAG_TAGS = "wcag2a,wcag2aa,wcag21a,wcag21aa,wcag22aa"

LAYOUT_CHECK = """
(() => {
    const errors = [];
    const visible = e => e.getBoundingClientRect().width > 0 && getComputedStyle(e).visibility !== 'hidden';
    const check = (condition, message) => { if (!condition) errors.push(message); };
    const panel = document.querySelector('[role="tabpanel"]:not([style*="display: none"])');
    const tabs = [...document.querySelectorAll('#studio-tabs [role="tab"]')].filter(visible);
    check(tabs.length === 3, `Only ${tabs.length} mode tabs are visible`);
    const wrapper = document.querySelector('#studio-tabs .tab-wrapper').getBoundingClientRect();
    for (const tab of tabs) {
        const r = tab.getBoundingClientRect();
        check(r.top >= wrapper.top && r.bottom <= wrapper.bottom, `Clipped tab: ${tab.textContent}`);
        check(r.left >= wrapper.left && r.right <= wrapper.right, `Overflowing tab: ${tab.textContent}`);
        check(Boolean(tab.getAttribute('aria-controls')), 'Missing tab panel ID');
        if (tab.getAttribute('aria-selected') === 'true') {
            check(Boolean(document.getElementById(tab.getAttribute('aria-controls'))), 'Missing active tab panel association');
        }
    }
    check(document.documentElement.scrollWidth <= innerWidth, 'Page scrolls horizontally');
    for (const el of [...panel.querySelectorAll('button, textarea, input, .studio-reference')].filter(visible)) {
        const r = el.getBoundingClientRect();
        check(r.left >= 0 && r.right <= innerWidth + 1, `Control outside viewport: ${el.id || el.outerHTML.slice(0, 100)}`);
        const player = el.closest('.studio-player, .studio-reference');
        if (player) {
            const bounds = player.getBoundingClientRect();
            check(r.left >= bounds.left - 1 && r.right <= bounds.right + 1, 'Audio control clipped by its player');
        }
    }
    for (const pill of [...panel.querySelectorAll('.preset-pill')].filter(visible)) {
        const range = document.createRange();
        range.selectNodeContents(pill);
        const text = range.getBoundingClientRect();
        const r = pill.getBoundingClientRect();
        check(text.height <= parseFloat(getComputedStyle(pill).lineHeight) + 1, `Wrapped preset: ${pill.textContent}`);
        check(r.height >= 44 && r.height <= 46, `Uneven preset height: ${pill.textContent} ${r.height}`);
        check(text.left >= r.left && text.right <= r.right, `Clipped preset label: ${pill.textContent}`);
    }
    for (const button of document.querySelectorAll('.studio-download')) {
        const range = document.createRange();
        range.selectNodeContents(button);
        const text = range.getBoundingClientRect();
        const r = button.getBoundingClientRect();
        check(text.left >= r.left && text.right <= r.right, 'Clipped Download WAV label');
        check(r.left >= 0 && r.right <= innerWidth, 'Download button outside viewport');
        const player = button.parentElement.querySelector('.studio-player').getBoundingClientRect();
        if (innerWidth <= 640) {
            check(r.top >= player.bottom, 'Mobile download button overlaps player');
        } else {
            check(r.left >= player.right, 'Download button overlaps player');
        }
    }
    for (const settings of panel.querySelectorAll('.studio-settings')) {
        const fields = [...settings.querySelectorAll('.studio-field')].map(e => e.getBoundingClientRect());
        const columns = settings.classList.contains('studio-settings-three') && innerWidth > 640 ? 3 : 2;
        const expected = (settings.getBoundingClientRect().width - (columns - 1) * parseFloat(getComputedStyle(settings).columnGap)) / columns;
        check(Math.abs(fields[0].width - expected) < 1, 'Settings fields do not fill their grid columns');
        check(Math.abs(fields[0].top - fields[1].top) < 1, 'Settings fields unexpectedly stack inside one column');
        if (fields.length === 3 && columns === 2) {
            check(fields[2].top >= fields[0].bottom, 'Mobile model selector does not reflow to its own row');
            check(Math.abs(fields[2].width - settings.getBoundingClientRect().width) < 1, 'Mobile model selector is not full-width');
        }
    }
    const [inputs, preview] = [...panel.querySelectorAll('.studio-workspace > .column')].map(e => e.getBoundingClientRect());
    if (innerWidth <= 900) {
        check(preview.top >= inputs.bottom, 'Mobile preview overlaps or sits alongside the form');
        check(Math.abs(preview.width - inputs.width) < 1, 'Mobile panels have unequal widths');
    } else {
        check(preview.left >= inputs.right, 'Desktop panels overlap');
    }
    const emptyFrames = [...panel.querySelectorAll('.studio-empty')].filter(visible)
        .filter(el => getComputedStyle(el).borderTopStyle === 'dashed');
    check(emptyFrames.length === (panel.querySelector('.studio-player audio') ? 0 : 1), 'Empty preview has duplicated or missing frames');
    const header = document.querySelector('#studio-header').getBoundingClientRect();
    const history = document.querySelector('#recent-generations').getBoundingClientRect();
    check(Math.abs(header.left - inputs.left) < 1, 'Header and form gutters do not align');
    check(Math.abs(history.left - inputs.left) < 1, 'History and form gutters do not align');
    return errors;
})();
"""


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [320, 375, 768, 1440])
def test_modes_reflow_without_clipping(browser, theme, width):
    browser.open(theme, width)
    for mode, panel_id in MODES.items():
        browser.click(f"#{panel_id}-button")
        browser("wait", "--fn", f"document.querySelector('#{mode}-controls').getBoundingClientRect().width > 0")
        errors = browser.evaluate(LAYOUT_CHECK)
        assert not errors, f"{theme}, {width}px, {mode}: {errors}"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_modes_have_no_wcag_a_or_aa_violations(browser, theme):
    browser.open(theme)
    for panel_id in MODES.values():
        browser.click(f"#{panel_id}-button")
        report = browser("a11y", "--tags", WCAG_TAGS)
        assert report["violations"] == [], (theme, panel_id, report["violations"])


def test_presets_populate_editable_instructions(browser):
    browser.open()
    for mode, presets in (("design", app.DESIGN_PRESETS), ("custom", app.STYLE_PRESETS)):
        browser.click(f"#{MODES[mode]}-button")
        for index, prompt in enumerate(presets.values(), start=1):
            browser.click(f"#{mode}-controls .preset-pill:nth-child({index})")
            expected = json.dumps(prompt)
            browser(
                "wait", "--fn",
                f"document.querySelector('#{mode}-instruction textarea').value === {expected}",
            )
        browser("fill", f"#{mode}-instruction textarea", "My own voice direction")
        assert browser.evaluate(
            f"document.querySelector('#{mode}-instruction textarea').value"
        ) == "My own voice direction"


def test_keyboard_mode_navigation_and_dropdown_selection(browser):
    browser.open(width=375)
    browser("wait", "--fn", "document.querySelector('#studio-tabs').dataset.keyboardReady === 'true'")
    browser("focus", "#voice-design-button")
    for key, expected in (("ArrowRight", "voice-clone"), ("End", "custom-voice"), ("Home", "voice-design"), ("ArrowLeft", "custom-voice")):
        browser("press", key)
        browser("wait", "--fn", f"document.querySelector('#{expected}-button').getAttribute('aria-selected') === 'true'")
        assert browser.evaluate("document.activeElement.id") == f"{expected}-button"
    browser("press", "Tab")
    assert browser.evaluate("document.activeElement.closest('#custom-text') !== null")
    for label, choice in (("Preset Speaker", "Dylan"), ("Language", "English"), ("Model Size", "0.6B")):
        selector = f'#custom-controls input[aria-label="{label}"]'
        browser.click(selector)
        browser("find", "role", "option", "click", "--name", choice, "--exact")
        assert browser.evaluate(f"document.querySelector({json.dumps(selector)}).value") == choice
    assert browser.evaluate(LAYOUT_CHECK) == []


def test_reduced_motion_disables_control_transitions(browser):
    browser("set", "media", "light", "reduced-motion")
    browser.open("light", 375)
    assert browser.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
    assert browser.evaluate(
        "getComputedStyle(document.querySelector('#design-generate')).transitionDuration"
    ) == "0s"
    assert browser.evaluate(
        "getComputedStyle(document.querySelector('.preset-pill')).transitionDuration"
    ) == "0s"


@pytest.mark.parametrize("width", [320, 1440])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_reference_history_errors_and_wav_downloads(browser, tmp_path, width, theme):
    browser.open(theme, width)
    browser.click("#design-generate")
    browser("wait", "--fn", "document.querySelectorAll('.history-card').length === 1")
    media = "document.querySelector('#design-audio #waveform > div')?.shadowRoot?.querySelector('audio')"
    browser("wait", "--fn", f"{media}?.readyState > 0")
    browser.click('#design-audio button[aria-label="Play"]')
    browser("wait", "--fn", f"{media}.currentTime > 0")
    assert browser.evaluate(
        "getComputedStyle(document.querySelector('#design-preview .studio-empty')).display"
    ) == "none"
    browser("fill", "#design-text textarea", "")
    browser.click("#design-generate")
    browser("wait", "--fn", "document.querySelector('#design-status [data-error=\"true\"]') !== null")
    assert browser.evaluate("document.querySelectorAll('.history-card').length") == 1
    assert browser.evaluate("document.querySelector('#design-audio audio') !== null")

    browser.click("#voice-clone-button")
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(72000, dtype=np.float32), 24000)
    browser("upload", "#clone-reference input[type=file]", str(reference))
    browser("wait", "--fn", "document.querySelector('#clone-reference-info').textContent.includes('3.00 s')")
    assert "24,000 Hz" in browser.evaluate("document.querySelector('#clone-reference-info').textContent")
    reference_root = "document.querySelector('#clone-reference #waveform > div')?.shadowRoot"
    browser("wait", "--fn", f"{reference_root}?.querySelector('audio')?.readyState > 0")
    browser.click('#clone-reference button[aria-label="Trim audio to selection"]')
    browser("wait", "--fn", f"{reference_root}?.querySelector('[part~=region-handle-right]') !== null")
    browser("scrollintoview", "#clone-reference .waveform-container")
    bounds = browser.evaluate(
        f"""(() => {{
            const root = {reference_root};
            return Object.fromEntries(['region', 'region-handle-right', 'wrapper'].map(part =>
                [part, root.querySelector(`[part~="${{part}}"]`).getBoundingClientRect().toJSON()]));
        }})()"""
    )
    handle, wave, region = bounds["region-handle-right"], bounds["wrapper"], bounds["region"]
    start = handle["x"] + handle["width"] / 2
    end = wave["x"] + wave["width"] * 0.75 - (region["right"] - start)
    y = handle["y"] + handle["height"] / 2
    browser("mouse", "move", str(start), str(y))
    browser("mouse", "down", "left")
    browser("mouse", "move", str(end), str(y), "--duration", "250", "--steps", "24")
    browser("mouse", "up", "left")
    browser("find", "role", "button", "click", "--name", "Trim", "--exact")
    browser("wait", "--fn", "!document.querySelector('#clone-reference-info').textContent.includes('3.00 s')")
    trimmed_src = browser.evaluate(f"{reference_root}.querySelector('audio').src")
    with urlopen(trimmed_src, timeout=5) as response:
        trimmed, sample_rate = sf.read(io.BytesIO(response.read()))
    assert len(trimmed) / sample_rate == pytest.approx(1.5, abs=0.04)
    metadata = browser.evaluate("document.querySelector('#clone-reference-info').textContent")
    assert f"{len(trimmed) / sample_rate:.2f} s" in metadata
    assert f"{sample_rate:,} Hz" in metadata
    assert browser.evaluate(LAYOUT_CHECK) == []
    assert browser("a11y", "--tags", WCAG_TAGS)["violations"] == []
    browser("check", "#clone-xvector input")
    browser.click("#clone-generate")
    browser("wait", "--fn", "document.querySelectorAll('.history-card').length === 2")
    browser.click('#clone-reference button[aria-label="Clear"]')
    browser("wait", "--fn", "document.querySelector('#clone-reference-info').textContent.includes('Upload or record')")
    browser.click("#clone-generate")
    browser("wait", "--fn", "document.querySelector('#clone-status [data-error=\"true\"]') !== null")
    assert browser.evaluate("document.querySelectorAll('.history-card').length") == 2
    browser.click("#custom-voice-button")
    browser.click("#custom-generate")
    browser("wait", "--fn", "document.querySelectorAll('.history-card').length === 3")
    assert browser.evaluate(
        "[...document.querySelectorAll('.mode-badge')].map(e => e.textContent)"
    ) == ["Custom Voice", "Voice Clone", "Voice Design"]
    downloaded = tmp_path / "downloaded.wav"
    browser("scrollintoview", ".history-card .studio-download")
    browser("download", ".history-card .studio-download", str(downloaded))
    data, sample_rate = sf.read(downloaded)
    assert sample_rate == 24000
    assert len(data) == 30000
    assert browser.evaluate(LAYOUT_CHECK) == []
    assert browser("a11y", "--tags", WCAG_TAGS)["violations"] == []
    browser.open()
    browser("wait", "--fn", "document.querySelector('.history-empty') !== null")
    assert browser.evaluate("document.querySelectorAll('.history-card').length") == 0
