from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from offscreen.algo.compile import compile_timeline
from offscreen.domain.plan import (
    AudioRef,
    Clip,
    EditPlan,
    PlanSegment,
    ScriptRef,
    SourceAudio,
    VoiceSpec,
)
from offscreen.domain.timeline import OutputSpec, Timeline
from offscreen.engine import ArtifactStore, Engine, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.stages.analysis.ingest import ingest
from offscreen.stages.output.compile import TIMELINE_FILE
from offscreen.stages.output.render import FINAL_FILE, SUBTITLES_FILE, RenderError, RenderStage
from offscreen.store.db import Database
from offscreen.store.files import write_model
from offscreen.store.repos import AssetRepo


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[AssetRepo]:
    db = Database(tmp_path / "db" / "o.db")
    yield AssetRepo(db)
    db.close()


def register(repo: AssetRepo, clip: Path, tmp_path: Path) -> str:
    movie = tmp_path / "movie" / "film.mp4"
    movie.parent.mkdir()
    movie.write_bytes(clip.read_bytes())
    return ingest(movie, repo).asset.id  # real ffprobe


class StubCompile(Stage):
    """Stands in for the compile stage: a timeline over a 3 s film, plus its narration."""

    name = "output.compile"
    version = 1
    lane = "cpu"

    def __init__(self, repo: AssetRepo, subtitles: bool = True, layout: str = "keep") -> None:
        self.repo, self.subtitles, self.layout = repo, subtitles, layout

    def params(self, scope):  # type: ignore[no-untyped-def]
        return {"subtitles": self.subtitles, "layout": self.layout}

    def run(self, ctx: StageContext) -> StageOutput:
        aid = ctx.scope["asset_id"]
        asset = self.repo.get(aid)
        assert asset is not None
        tts = FakeTTS(chars_per_s=5.0)
        segs = []
        (ctx.out_dir / "tts").mkdir()
        for i, (text, clips) in enumerate(
            [("你好，世界。", [(0, 1000), (1000, 2000)]), ("再见了朋友", [(2000, 3000)])], 1
        ):
            audio = tts.synthesize("字" * (10 if i == 1 else 5), voice_id="v")
            (ctx.out_dir / f"tts/{i}.wav").write_bytes(audio.data)
            segs.append(
                PlanSegment(
                    id=f"seg_{i:02d}",
                    kind="narration",
                    text=text,
                    voice=VoiceSpec(voice_id="v"),
                    audio=AudioRef(
                        file=f"tts/{i}.wav",
                        duration_ms=audio.duration_ms,
                        char_timings=[(k * 100, (k + 1) * 100) for k in range(len(text))],
                    ),
                    clips=[Clip(asset_id=aid, src_in_ms=a, src_out_ms=b) for a, b in clips],
                    source_audio=SourceAudio(mode="duck", gain_db=-20.0),
                )
            )
        plan = EditPlan(
            id="pln_t",
            project_id="prj_t",
            version=1,
            author="ai",
            script_ref=ScriptRef(id="scr_t", version=1),
            segments=segs,
        )
        spec = OutputSpec(
            profile="source",
            width=asset.video.width,
            height=asset.video.height,
            fps=asset.video.fps,
            layout=self.layout,
        )  # type: ignore[arg-type]
        tl = compile_timeline(plan, spec, subtitle_max_chars=22)
        if not self.subtitles:
            tl = tl.model_copy(update={"subtitles": []})
        write_model(ctx.out_dir / TIMELINE_FILE, tl)
        return StageOutput()


def ffprobe(path: Path) -> dict:  # type: ignore[type-arg]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-print_format", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)  # type: ignore[no-any-return]


def frame_gray(path: Path, at_s: float) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(at_s), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip


def mean_volume_db(path: Path) -> float:
    r = subprocess.run(["ffmpeg", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)  # fmt: skip
    m = re.search(r"max_volume: (-?[\d.]+) dB", r.stderr)
    assert m, r.stderr[-400:]
    return float(m.group(1))


def render(repo: AssetRepo, tmp_path: Path, aid: str, name: str, **kw: object):  # type: ignore[no-untyped-def]
    engine = Engine(
        ArtifactStore(tmp_path / name),
        [StubCompile(repo, **kw), RenderStage(repo)],  # type: ignore[arg-type]
    )
    return engine.ensure("output.render", {"asset_id": aid})


def test_render_produces_the_timeline_exactly(clip: Path, repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, clip, tmp_path)
    art = render(repo, tmp_path, aid, "a")
    tl = (
        Engine(ArtifactStore(tmp_path / "a"), [StubCompile(repo)])
        .ensure("output.compile", {"asset_id": aid})
        .read_model(TIMELINE_FILE, Timeline)
    )

    info = ffprobe(art.path(FINAL_FILE))
    kinds = {s["codec_type"]: s for s in info["streams"]}
    # Every frame of the timeline, at the film's size and rate, with sound.
    assert kinds["video"]["nb_read_frames"] == str(tl.duration_frames)
    assert (kinds["video"]["width"], kinds["video"]["height"]) == (320, 180)
    assert kinds["video"]["r_frame_rate"] == "24000/1001"
    assert kinds["audio"]["codec_name"] == "aac"
    expected_ms = tl.duration_frames * 1001 / 24
    assert abs(float(info["format"]["duration"]) * 1000 - expected_ms) < 80
    # The film's own sound is under the narration (the narration here is silent).
    assert -60 < mean_volume_db(art.path(FINAL_FILE)) < -20

    assert art.path(SUBTITLES_FILE).read_text(encoding="utf-8").count("Dialogue:") == len(
        tl.subtitles
    )
    assert art.meta["duration_frames"] == tl.duration_frames and art.meta["subtitles"] == len(
        tl.subtitles
    )
    # Scratch files are gone: only the final video and the subtitles remain.
    assert sorted(f.name for f in art.dir.iterdir() if f.name != "manifest.json") == [
        FINAL_FILE,
        SUBTITLES_FILE,
    ]


def test_subtitles_are_burned_into_the_lower_part_of_the_picture(
    clip: Path, repo: AssetRepo, tmp_path: Path
) -> None:
    aid = register(repo, clip, tmp_path)
    with_subs = render(repo, tmp_path, aid, "s", subtitles=True).path(FINAL_FILE)
    without = render(repo, tmp_path, aid, "n", subtitles=False).path(FINAL_FILE)
    w, h = 320, 180
    a, b = frame_gray(with_subs, 0.3), frame_gray(without, 0.3)
    assert len(a) == len(b) == w * h
    top = sum(abs(x - y) > 30 for x, y in zip(a[: w * h // 2], b[: w * h // 2], strict=True))
    bottom = sum(abs(x - y) > 30 for x, y in zip(a[w * h // 2 :], b[w * h // 2 :], strict=True))
    assert bottom > 100 and bottom > 10 * max(top, 1)


def test_missing_source_is_reported(clip: Path, repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, clip, tmp_path)
    (tmp_path / "movie" / "film.mp4").unlink()
    with pytest.raises(RenderError, match="not found"):
        render(repo, tmp_path, aid, "m")


def test_unsupported_layout_is_rejected(clip: Path, repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, clip, tmp_path)
    with pytest.raises(RenderError, match="layout"):
        render(repo, tmp_path, aid, "l", layout="blur_pad")


def test_cancel_stops_the_render(clip: Path, repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, clip, tmp_path)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    engine = Engine(
        ArtifactStore(tmp_path / "c"), [StubCompile(repo), RenderStage(repo)], is_canceled=cancel
    )
    with pytest.raises(StageCanceled):
        engine.ensure("output.render", {"asset_id": aid})
    assert not list((tmp_path / "c").glob("output.render/*/manifest.json"))


def test_cache_key_covers_the_source_film(clip: Path, repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, clip, tmp_path)
    p = RenderStage(repo).params({"asset_id": aid})
    assert p["source_fingerprint"].startswith("sha256:") and p["font"] and p["final_crf"]
    with pytest.raises(RenderError, match="unknown asset"):
        RenderStage(repo).params({"asset_id": "ast_nope"})
