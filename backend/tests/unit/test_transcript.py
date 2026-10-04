from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from offscreen.domain.index import Transcript, Word
from offscreen.engine import ArtifactStore, Engine, StageCanceled
from offscreen.providers.adapters.faster_whisper_asr import FasterWhisperAsr
from offscreen.providers.ports import AsrCanceled, AsrResult, AsrSegment
from offscreen.stages.analysis.ingest import ingest
from offscreen.stages.analysis.proxy import ProxyStage
from offscreen.stages.analysis.transcript import (
    TRANSCRIPT_FILE,
    TranscriptError,
    TranscriptStage,
    decode_subtitle_bytes,
)
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo

SRT = (
    "1\n00:00:00,500 --> 00:00:01,500\nHello there\n\n"
    "2\n00:00:01,600 --> 00:00:02,400\nGeneral Kenobi\n"
)


class FakeAsr:
    id = "fake-asr@1"

    def __init__(self) -> None:
        self.calls = 0
        self.wavs: list[Path] = []

    def transcribe(
        self,
        wav: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsrResult:
        self.calls += 1
        self.wavs.append(wav)
        if on_progress:
            on_progress(0.5)
        # Deliberately unordered: the stage must sort.
        return AsrResult(
            "en",
            [
                AsrSegment(1000, 2000, "second", [Word(w="second", start_ms=1000, end_ms=2000)]),
                AsrSegment(0, 900, "first"),
            ],
        )


def build(tmp_path: Path, asr: Any, **kw: Any) -> tuple[AssetRepo, Engine]:
    repo = AssetRepo(Database(tmp_path / "db" / "o.db"))
    engine = Engine(
        ArtifactStore(tmp_path / "artifacts"), [ProxyStage(repo), TranscriptStage(repo, asr)], **kw
    )
    return repo, engine


def copy_clip(clip: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "movie" / "film.mp4"
    dst.parent.mkdir()
    dst.write_bytes(clip.read_bytes())
    return dst


def test_external_srt_wins_and_asr_is_not_called(clip: Path, tmp_path: Path) -> None:
    movie = copy_clip(clip, tmp_path)
    movie.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    asr = FakeAsr()
    repo, engine = build(tmp_path, asr)
    asset = ingest(movie, repo).asset

    art = engine.ensure("analysis.transcript", {"asset_id": asset.id})
    doc = art.read_model(TRANSCRIPT_FILE, Transcript)
    assert (doc.source, doc.language) == ("subtitle:external", "und")
    assert [(x.id, x.start_ms, x.end_ms, x.text) for x in doc.lines] == [
        ("ln_0001", 500, 1500, "Hello there"),
        ("ln_0002", 1600, 2400, "General Kenobi"),
    ]
    assert asr.calls == 0
    assert art.meta == {"lines": 2, "source": "subtitle:external"}


def test_editing_the_subtitle_file_invalidates_the_cache(clip: Path, tmp_path: Path) -> None:
    movie = copy_clip(clip, tmp_path)
    srt = movie.with_suffix(".srt")
    srt.write_text(SRT, encoding="utf-8")
    repo, engine = build(tmp_path, None)
    asset = ingest(movie, repo).asset
    first = engine.ensure("analysis.transcript", {"asset_id": asset.id})
    assert engine.ensure("analysis.transcript", {"asset_id": asset.id}).dir == first.dir

    srt.write_text(SRT.replace("Hello", "Howdy"), encoding="utf-8")
    second = engine.ensure("analysis.transcript", {"asset_id": asset.id})
    assert second.dir != first.dir
    assert second.read_model(TRANSCRIPT_FILE, Transcript).lines[0].text == "Howdy there"


def test_gbk_encoded_subtitles_are_decoded(clip: Path, tmp_path: Path) -> None:
    movie = copy_clip(clip, tmp_path)
    movie.with_suffix(".srt").write_bytes(
        "1\n00:00:00,000 --> 00:00:01,000\n你好，世界\n".encode("gb18030")
    )
    repo, engine = build(tmp_path, None)
    asset = ingest(movie, repo).asset
    doc = engine.ensure("analysis.transcript", {"asset_id": asset.id}).read_model(
        TRANSCRIPT_FILE, Transcript
    )
    assert doc.lines[0].text == "你好，世界" and doc.language == "zh"


def test_decode_handles_utf16_and_bom() -> None:
    assert decode_subtitle_bytes("héllo".encode("utf-16")) == "héllo"
    assert decode_subtitle_bytes(b"\xef\xbb\xbfabc") == "abc"


def test_without_subtitles_asr_runs_on_the_16k_wav_and_output_is_sorted(
    clip: Path, tmp_path: Path
) -> None:
    asr = FakeAsr()
    repo, engine = build(tmp_path, asr)
    asset = ingest(clip, repo).asset
    art = engine.ensure("analysis.transcript", {"asset_id": asset.id})

    doc = art.read_model(TRANSCRIPT_FILE, Transcript)
    assert doc.source == "asr:fake-asr@1" and doc.language == "en"
    assert [(x.id, x.text) for x in doc.lines] == [("ln_0001", "first"), ("ln_0002", "second")]
    assert doc.lines[1].words[0].w == "second"
    with wave.open(str(asr.wavs[0])) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)

    engine.ensure("analysis.transcript", {"asset_id": asset.id})
    assert asr.calls == 1  # cache hit


def test_changing_the_asr_engine_invalidates_the_cache(clip: Path, tmp_path: Path) -> None:
    asr = FakeAsr()
    repo, engine = build(tmp_path, asr)
    asset = ingest(clip, repo).asset
    engine.ensure("analysis.transcript", {"asset_id": asset.id})
    asr.id = "fake-asr@2"
    engine.ensure("analysis.transcript", {"asset_id": asset.id})
    assert asr.calls == 2


def test_no_subtitles_and_no_asr_is_a_clear_error(clip: Path, tmp_path: Path) -> None:
    repo, engine = build(tmp_path, None)
    asset = ingest(clip, repo).asset
    with pytest.raises(TranscriptError, match="no external subtitle"):
        engine.ensure("analysis.transcript", {"asset_id": asset.id})


def test_empty_subtitle_file_falls_back_to_asr(clip: Path, tmp_path: Path) -> None:
    movie = copy_clip(clip, tmp_path)
    movie.with_suffix(".srt").write_text("", encoding="utf-8")
    asr = FakeAsr()
    repo, engine = build(tmp_path, asr)
    asset = ingest(movie, repo).asset
    engine.ensure("analysis.transcript", {"asset_id": asset.id})
    assert asr.calls == 1


def test_unparseable_subtitle_is_an_error(clip: Path, tmp_path: Path) -> None:
    movie = copy_clip(clip, tmp_path)
    movie.with_suffix(".ass").write_text("[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,,hi\n")
    repo, engine = build(tmp_path, FakeAsr())
    asset = ingest(movie, repo).asset
    with pytest.raises(TranscriptError, match="cannot parse"):
        engine.ensure("analysis.transcript", {"asset_id": asset.id})


def test_silent_video_yields_an_empty_transcript(tmp_path: Path) -> None:
    import subprocess

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=160x90:rate=24:duration=1", "-an", "-c:v", "libx264", str(silent)],
        check=True,
    )  # fmt: skip
    asr = FakeAsr()
    repo, engine = build(tmp_path, asr)
    asset = ingest(silent, repo).asset
    doc = engine.ensure("analysis.transcript", {"asset_id": asset.id}).read_model(
        TRANSCRIPT_FILE, Transcript
    )
    assert (doc.lines, doc.source, asr.calls) == ([], "none", 0)


def test_asr_cancellation_becomes_stage_canceled(clip: Path, tmp_path: Path) -> None:
    class Canceling(FakeAsr):
        def transcribe(self, wav: Path, **_: Any) -> AsrResult:
            raise AsrCanceled("stop")

    repo, engine = build(tmp_path, Canceling())
    asset = ingest(clip, repo).asset
    with pytest.raises(StageCanceled):
        engine.ensure("analysis.transcript", {"asset_id": asset.id})


# ---- faster-whisper adapter, with an injected fake engine -----------------------------
class FakeWhisper:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
        self.kwargs = kwargs
        w = SimpleNamespace
        segs = [
            w(
                start=0.1234, end=1.5, text=" Hello world ",
                words=[
                    w(word=" Hello", start=0.1234, end=0.6),
                    w(word=" world", start=0.6, end=0.6),
                ],
            ),
            w(start=2.0, end=2.0, text="zero length", words=None),
            w(start=3.0, end=4.0, text="   ", words=[]),
        ]  # fmt: skip
        return iter(segs), w(language="en", duration=5.0)


def test_faster_whisper_adapter_converts_segments() -> None:
    fake = FakeWhisper()
    asr = FasterWhisperAsr("tiny", device="cpu", engine=fake)
    seen: list[float] = []
    res = asr.transcribe(Path("x.wav"), on_progress=seen.append)

    assert res.language == "en"
    assert [(s.start_ms, s.end_ms, s.text) for s in res.segments] == [
        (123, 1500, "Hello world"),
        (2000, 2001, "zero length"),  # end clamped so the range stays valid
    ]
    assert [(x.w, x.start_ms, x.end_ms) for x in res.segments[0].words] == [
        ("Hello", 123, 600),
        ("world", 600, 601),
    ]
    assert fake.kwargs["word_timestamps"] is True and fake.kwargs["vad_filter"] is True
    assert seen == sorted(seen) and seen[-1] == pytest.approx(0.8)
    assert asr.id == "faster-whisper/tiny@1"


def test_faster_whisper_adapter_cancels_between_segments() -> None:
    asr = FasterWhisperAsr("tiny", device="cpu", engine=FakeWhisper())
    with pytest.raises(AsrCanceled):
        asr.transcribe(Path("x.wav"), should_cancel=lambda: True)


def test_faster_whisper_missing_package_has_a_helpful_error() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="not installed"):
            FasterWhisperAsr("tiny", device="cpu").transcribe(Path("x.wav"))
