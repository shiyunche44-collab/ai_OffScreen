"""The voice library: calibration, registry, and what the pipeline does with a measured voice."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from typer.testing import CliRunner

from offscreen.algo.calibrate import (
    STANDARD_TEXTS,
    TARGET_SPREAD,
    fit_rate,
    voice_timing,
)
from offscreen.algo.script import DEFAULT_CHARS_PER_S, count_chars, estimate_duration_s
from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.domain.voice import Voice, VoiceLibrary
from offscreen.providers.adapters.fake import FakeFaceAnalyzer, FakeLLM, FakeTTS
from offscreen.providers.ports import TTSAuthError
from offscreen.services.app import AppServices
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers, RunOptions
from offscreen.stages.creation.script import ScriptSettings
from offscreen.store.voices import VoiceStore

NOW = datetime(2026, 10, 10, tzinfo=UTC)
FAKE_RATE = 5.5
"""FakeTTS speaks every character that is not a space at this rate, punctuation included, so
the rate over spoken characters (letters and digits) is a little lower."""
FAKE_MEASURED = (
    FAKE_RATE
    * sum(map(count_chars, STANDARD_TEXTS))
    / sum(len("".join(t.split())) for t in STANDARD_TEXTS)
)


def measured(**over: object) -> Voice:
    base: dict[str, object] = {
        "id": "v1",
        "name": "v1",
        "provider": "fake",
        "chars_per_s": 5.0,
        "rate_spread": 0.01,
        "calibrated_with": "tts-a",
        "calibrated_at": NOW,
    }
    return Voice.model_validate(base | over)


# ---- algo ----------------------------------------------------------------------------------
def test_the_rate_is_all_characters_over_all_seconds() -> None:
    fit = fit_rate([(100, 20_000), (50, 10_000)])
    assert fit.chars_per_s == pytest.approx(5.0) and fit.spread == pytest.approx(0)
    # a long text counts for more than a short one
    assert fit_rate([(90, 20_000), (10, 5_000)]).chars_per_s == pytest.approx(100 / 25)


def test_the_spread_is_how_far_a_sample_strays_from_the_rate() -> None:
    fit = fit_rate([(100, 20_000), (110, 20_000)])  # 5.0 and 5.5 around 5.2381
    assert fit.spread == pytest.approx(abs(5.5 - fit.chars_per_s) / fit.chars_per_s)


def test_no_samples_or_empty_ones_are_refused() -> None:
    for bad in ([], [(0, 1000)], [(10, 0)]):
        with pytest.raises(ValueError):
            fit_rate(bad)


@given(st.lists(st.tuples(st.integers(1, 2000), st.integers(100, 600_000)), min_size=1, max_size=8))
def test_the_rate_lies_among_the_samples_and_predicts_the_total(
    samples: list[tuple[int, int]],
) -> None:
    fit = fit_rate(samples)
    rates = [c / (ms / 1000) for c, ms in samples]
    assert min(rates) - 1e-9 <= fit.chars_per_s <= max(rates) + 1e-9
    assert fit.spread >= 0
    total_s = sum(ms for _, ms in samples) / 1000
    estimated = estimate_duration_s(sum(c for c, _ in samples), fit.chars_per_s)
    assert estimated == pytest.approx(total_s, rel=1e-9)  # the whole is predicted exactly


def test_the_standard_texts_are_substantial_and_all_spoken_characters() -> None:
    assert len(STANDARD_TEXTS) >= 3
    assert all(60 <= count_chars(t) <= 120 for t in STANDARD_TEXTS)
    assert not any(c.isascii() and c.isalnum() for t in STANDARD_TEXTS for c in t)


def test_timing_uses_the_measured_rate_only_for_the_engine_that_measured_it() -> None:
    v = measured(default_speed=1.1)
    t = voice_timing(v, "tts-a")
    assert (t.measured, t.speed) == (True, 1.1) and t.chars_per_s == pytest.approx(5.5)
    other = voice_timing(v, "tts-b")
    assert other.measured is False
    assert other.chars_per_s == pytest.approx(DEFAULT_CHARS_PER_S * 1.1)
    plain = voice_timing(None, "tts-a")
    assert (plain.measured, plain.speed, plain.chars_per_s) == (False, 1.0, DEFAULT_CHARS_PER_S)


# ---- domain --------------------------------------------------------------------------------
def test_a_calibration_is_whole_or_absent() -> None:
    Voice(id="a", name="a", provider="p")
    with pytest.raises(ValidationError, match="calibration"):
        Voice(id="a", name="a", provider="p", chars_per_s=4.0)


def test_voice_ids_are_unique_and_speeds_bounded() -> None:
    a = Voice(id="a", name="a", provider="p")
    with pytest.raises(ValidationError, match="duplicate"):
        VoiceLibrary(voices=[a, a])
    with pytest.raises(ValidationError):
        Voice(id="a", name="a", provider="p", default_speed=3)


# ---- service -------------------------------------------------------------------------------
@pytest.fixture
def services(tmp_path: Path) -> Iterator[AppServices]:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    fakes = Providers(
        llm=FakeLLM({}),
        tts=FakeTTS(chars_per_s=FAKE_RATE),
        detector=None,  # type: ignore[arg-type]
        faces=FakeFaceAnalyzer(),
    )
    with AppServices(cfg, providers=fakes) as s:
        yield s


def test_calibration_measures_the_rate_and_registers_the_voice(services: AppServices) -> None:
    assert services.voices.list() == []
    v = services.voices.calibrate("narrator")
    assert v.chars_per_s == pytest.approx(FAKE_MEASURED, rel=0.001)
    assert v.rate_spread is not None and v.rate_spread <= TARGET_SPREAD
    assert services.voices.accurate(v)
    assert (v.provider, v.calibrated_with) == ("edge_tts", services.jobs.providers.tts.id)
    assert services.voices.get("narrator") == v  # kept on disk
    tts = services.jobs.providers.tts
    assert isinstance(tts, FakeTTS)
    assert [c[1:] for c in tts.calls] == [("narrator", 1.0)] * len(STANDARD_TEXTS)


def test_the_estimate_for_a_text_is_within_five_percent_of_the_audio(
    services: AppServices,
) -> None:
    """The M6 exit criterion, on text the rate was not measured on."""
    v = services.voices.calibrate("narrator")
    text = "他们沿着河岸走了很久，谁也没有说话。直到天快黑的时候，远处终于出现了一点灯火。"
    audio = services.jobs.providers.tts.synthesize(text, voice_id="narrator", speed=1.0)
    estimated_ms = estimate_duration_s(count_chars(text), v.chars_per_s or 0) * 1000
    assert abs(estimated_ms - audio.duration_ms) / audio.duration_ms <= TARGET_SPREAD


def test_a_failed_synthesis_stores_nothing(services: AppServices) -> None:
    tts = services.jobs.providers.tts
    assert isinstance(tts, FakeTTS)
    tts.fail_with = TTSAuthError("no key")
    with pytest.raises(InvalidInput, match="no key"):
        services.voices.calibrate("narrator")
    assert services.voices.list() == []


def test_adding_changing_and_removing_voices(services: AppServices) -> None:
    store = VoiceStore(services.cfg.data_dir)
    store.dir.mkdir(parents=True)
    (store.dir / "me.wav").write_bytes(b"RIFF")
    v = services.voices.add("me", name="我", provider="cosyvoice", reference_audio="me.wav")
    assert (v.name, v.provider, v.default_speed) == ("我", "cosyvoice", 1.0)
    with pytest.raises(InvalidInput, match="no recording"):
        services.voices.add("me", reference_audio="missing.wav")
    with pytest.raises(InvalidInput):
        services.voices.add("me", default_speed=5)

    services.voices.calibrate("me")
    kept = services.voices.add("me", default_speed=1.2)  # a speed change keeps the measurement
    assert kept.default_speed == 1.2 and kept.chars_per_s is not None
    other = services.voices.add("me", provider="minimax")  # another engine: not the same voice
    assert other.chars_per_s is None and other.calibrated_with is None

    services.voices.add("second")
    assert [x.id for x in services.voices.list()] == ["me", "second"]
    services.voices.remove("me")
    assert [x.id for x in services.voices.list()] == ["second"]
    with pytest.raises(NotFound):
        services.voices.remove("me")
    with pytest.raises(NotFound):
        services.voices.get("me")


def test_the_script_is_sized_for_the_voice_that_reads_it(services: AppServices) -> None:
    def settings(voice: str | None = None) -> ScriptSettings:
        with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
            stage = next(
                s for s in p._stages(RunOptions(voice=voice)) if s.name == "creation.script"
            )
        assert isinstance(stage, object) and hasattr(stage, "settings")
        return stage.settings  # type: ignore[no-any-return]

    default = services.cfg.tts.default_voice
    assert settings().chars_per_s == DEFAULT_CHARS_PER_S  # nobody measured it
    services.voices.calibrate(default)
    services.voices.add(default, default_speed=1.1)
    fast = settings()
    assert fast.chars_per_s == pytest.approx(FAKE_MEASURED * 1.1, rel=0.001)
    assert settings("someone-else").chars_per_s == DEFAULT_CHARS_PER_S


# ---- command line --------------------------------------------------------------------------
def test_the_command_line_lists_adds_calibrates_and_removes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data_dir: {tmp_path / 'data'}\ntts:\n  provider: edge_tts\n  default_voice: narrator\n",
        encoding="utf-8",
    )
    run = CliRunner()

    def cli(*args: str) -> str:
        r = run.invoke(app, ["voice", *args, "--config", str(config)])
        assert r.exit_code == 0, r.output
        return r.output

    assert "no voices registered" in cli("list")
    out = cli("add", "narrator", "--name", "旁白", "--speed", "1.1")
    assert "narrator *" in out and "未标定" in out and "旁白" in out
    bad = run.invoke(app, ["voice", "add", "narrator", "--speed", "9", "--config", str(config)])
    assert bad.exit_code == 1
    assert "narrator" in cli("list")
    assert "removed narrator" in cli("remove", "narrator")
    assert "no voices registered" in cli("list")
