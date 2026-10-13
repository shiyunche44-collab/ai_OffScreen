from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from offscreen.algo.mixing import MixSettings, db_to_gain
from offscreen.domain.common import Rational
from offscreen.domain.timeline import (
    BgmItem,
    Duck,
    NarrationItem,
    OutputSpec,
    PlanRef,
    SourceAudioItem,
    Timeline,
    VideoItem,
)
from offscreen.media.audio import decode_audio, write_wav
from offscreen.stages.output.render import RenderError, mix_timeline

RATE = 48_000
FPS = Rational(num=25, den=1)


def tone(path: Path, seconds: float, value: float, rate: int = RATE) -> Path:
    """A wav of constant level (easy to measure after mixing)."""
    write_wav(path, np.full((round(seconds * rate), 1), value, dtype=np.float32), rate)
    return path


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return (data.reshape(-1, w.getnchannels()) / 32767.0).astype(np.float32)


def test_decode_is_exactly_as_long_as_asked(clip: Path) -> None:
    a = decode_audio(clip, sample_rate=RATE, channels=2, seek_ms=500, duration_ms=400)
    assert a.shape == (19_200, 2) and a.dtype == np.float32
    assert 0.05 < np.abs(a).max() <= 1.0  # the 440 Hz sine of the test clip
    # past the end of the file: padded with silence to the asked length
    tail = decode_audio(clip, sample_rate=RATE, channels=1, seek_ms=2800, duration_ms=1000)
    assert tail.shape == (48_000, 1) and np.all(tail[15_000:] == 0)
    whole = decode_audio(clip, sample_rate=16_000, channels=1)
    assert abs(len(whole) - 48_000) < 400


def test_write_wav_clamps(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    write_wav(p, np.array([[0.5, -2.0], [2.0, 0.0]], dtype=np.float32), 8000)
    back = read_wav(p)
    assert back.shape == (2, 2)
    assert np.allclose(back, [[0.5, -1.0], [1.0, 0.0]], atol=1e-4)


def timeline(n_frames: int, **tracks: object) -> Timeline:
    return Timeline(
        plan_ref=PlanRef(id="pln_t", version=1),
        output=OutputSpec(profile="t", width=320, height=180, fps=FPS),
        duration_frames=n_frames,
        video=[
            VideoItem(seg="seg_01", f0=0, f1=n_frames, asset_id="ast_t", src_in_ms=0, src_out_ms=1)
        ],
        **tracks,  # type: ignore[arg-type]
    )


def test_music_is_ducked_under_the_narration(tmp_path: Path) -> None:
    (tmp_path / "tts").mkdir()
    (tmp_path / "bgm").mkdir()
    tone(tmp_path / "tts" / "n.wav", 1.0, 0.1, rate=32_000)  # engines speak at other rates
    tone(tmp_path / "bgm" / "m.wav", 5.0, 0.5)
    tl = timeline(
        100,  # 4 s
        narration=[NarrationItem(seg="seg_01", f0=25, file="tts/n.wav")],  # 1 s .. 2 s
        bgm=[BgmItem(f0=0, f1=100, file="m.wav", gain_db=0.0, duck=Duck(depth_db=-20.0))],
    )
    mix = MixSettings(duck_attack_ms=100, duck_release_ms=200)
    report = mix_timeline(
        tl, tmp_path, tmp_path / "mix.wav", film_audio=None, bgm_dir=tmp_path / "bgm", mix=mix
    )
    out = read_wav(tmp_path / "mix.wav")
    assert out.shape == (4 * RATE, 2)  # exactly the programme
    at = lambda s: float(out[round(s * RATE), 0])  # noqa: E731
    assert at(0.5) == pytest.approx(0.5, abs=1e-3)  # music alone
    assert at(1.5) == pytest.approx(0.1 + 0.5 * db_to_gain(-20.0), abs=1e-3)  # under the voice
    assert 0.05 < at(0.95) < 0.5  # going down before the voice starts
    assert at(2.1) < at(2.19) < 0.5  # coming back after it
    assert at(3.0) == pytest.approx(0.5, abs=1e-3)
    assert report.clipped == 0 and report.peak == pytest.approx(0.5, abs=1e-3)


def test_film_sound_is_cut_to_its_items_and_missing_files_are_named(
    clip: Path, tmp_path: Path
) -> None:
    tl = timeline(
        50,
        source_audio=[
            SourceAudioItem(
                seg="seg_01", f0=0, f1=25, asset_id="ast_t", stem="mix", src_in_ms=0, gain_db=0
            )
        ],
    )
    mix_timeline(tl, tmp_path, tmp_path / "m.wav", film_audio=clip, bgm_dir=None, mix=MixSettings())
    out = read_wav(tmp_path / "m.wav")
    assert out.shape == (2 * RATE, 2)
    assert np.abs(out[: RATE - 1000]).max() > 0.05 and np.all(out[RATE:] == 0)

    with pytest.raises(RenderError, match="missing"):
        mix_timeline(
            timeline(50, narration=[NarrationItem(seg="seg_01", f0=0, file="nope.wav")]),
            tmp_path,
            tmp_path / "m.wav",
            film_audio=None,
            bgm_dir=None,
            mix=MixSettings(),
        )
    with pytest.raises(RenderError, match="music folder"):
        mix_timeline(
            timeline(50, bgm=[BgmItem(f0=0, f1=50, file="m.wav")]),
            tmp_path,
            tmp_path / "m.wav",
            film_audio=None,
            bgm_dir=None,
            mix=MixSettings(),
        )
