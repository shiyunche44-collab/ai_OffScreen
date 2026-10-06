"""Use cases of the CLI: import a movie, run one stage, run the whole chain to a video.

This is where stages meet providers: `Providers` bundles the model-backed ports (tests hand in
fakes), `Pipeline` builds the stage graph around them. Everything is cached by the engine, so
running the same command again reuses every artifact whose inputs did not change."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from offscreen.config import AppConfig, ConfigError
from offscreen.engine import Artifact, ArtifactStore, Engine, StageRun
from offscreen.engine.stage import Stage
from offscreen.log import current_job_id
from offscreen.media.ffmpeg import FFmpegCanceled, FFmpegError
from offscreen.media.probe import ProbeError
from offscreen.providers.adapters.faster_whisper_asr import FasterWhisperAsr
from offscreen.providers.adapters.insightface_faces import InsightFaceAnalyzer
from offscreen.providers.adapters.scenedetect_adapter import SceneDetectShots
from offscreen.providers.adapters.st_embedder import SentenceTransformerEmbedder
from offscreen.providers.adapters.transnetv2_shots import TransNetV2Shots
from offscreen.providers.ports import (
    ASR,
    LLM,
    TTS,
    Embedder,
    FaceAnalyzer,
    LLMError,
    ShotDetector,
    TTSError,
)
from offscreen.services.llm import build_llm, task_models
from offscreen.services.tts import build_tts
from offscreen.stages.analysis.captions import DEFAULT_BATCH, CaptionsError, CaptionsStage
from offscreen.stages.analysis.characters import CharactersError, CharactersStage
from offscreen.stages.analysis.embeddings import EmbeddingsError, EmbeddingsStage
from offscreen.stages.analysis.faces import FacesError, FacesStage
from offscreen.stages.analysis.ingest import IngestError, ingest
from offscreen.stages.analysis.keyframes import KeyframesStage
from offscreen.stages.analysis.naming import NamingStage
from offscreen.stages.analysis.proxy import ProxyError, ProxyStage
from offscreen.stages.analysis.scenes import ScenesError, ScenesStage
from offscreen.stages.analysis.shots import ShotsError, ShotsStage
from offscreen.stages.analysis.story import StoryError, StoryStage
from offscreen.stages.analysis.transcript import TranscriptError, TranscriptStage
from offscreen.stages.creation.outline import OutlineError, OutlineSettings, OutlineStage
from offscreen.stages.creation.plan import PlanError, PlanStage
from offscreen.stages.creation.script import ScriptError, ScriptSettings, ScriptStage
from offscreen.stages.output.compile import CompileStage, CompileStageError
from offscreen.stages.output.render import FINAL_FILE, RenderError, RenderStage
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, StageRunRepo

FINAL_STAGE = "output.render"
DEFAULT_STYLE = "neutral"

EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    ConfigError, IngestError, ProbeError, ProxyError, ShotsError, FacesError, CharactersError,
    EmbeddingsError, TranscriptError, StoryError, OutlineError, ScriptError, CaptionsError,
    ScenesError, PlanError, CompileStageError, RenderError, FFmpegError, FFmpegCanceled,
    LLMError, TTSError, ValueError,
)  # fmt: skip
"""Failures with a message meant for the person at the terminal (not bugs)."""

ProgressFn = Callable[[str, float, str], None]
ResolvedFn = Callable[[str, bool], None]


@dataclass
class Providers:
    llm: LLM
    tts: TTS
    detector: ShotDetector
    asr: ASR | None = None
    """None: the transcript stage needs an external subtitle file."""
    faces: FaceAnalyzer | None = None
    """None: the faces stage cannot run."""
    image_embedder: Embedder | None = None
    """Embeds keyframes and, into the same space, search queries."""
    text_embedder: Embedder | None = None
    """Embeds shot descriptions and search queries as text."""


@dataclass(frozen=True)
class RunOptions:
    minutes: float = 3.0
    voice: str | None = None
    """Voice id; None: the configured default."""
    style: str = DEFAULT_STYLE
    spoil_ending: bool = True


@dataclass
class StageReport:
    stage: str
    cached: bool


@dataclass
class RunResult:
    asset_id: str
    title: str
    artifact: Artifact
    stages: list[StageReport] = field(default_factory=list)

    @property
    def final(self) -> Path:
        return self.artifact.path(FINAL_FILE)


def build_providers(cfg: AppConfig, db: Database) -> Providers:
    asr: ASR | None = None
    if cfg.asr.provider == "faster_whisper":
        asr = FasterWhisperAsr(cfg.asr.model, device=cfg.asr.device)
    detector: ShotDetector = (
        TransNetV2Shots(threshold=cfg.shots.threshold, device=cfg.shots.device)
        if cfg.shots.detector == "transnetv2"
        else SceneDetectShots()
    )
    emb = cfg.embeddings
    image_embedder = (
        SentenceTransformerEmbedder(
            emb.image_text_model, image_model=emb.image_model, device=emb.device
        )
        if emb.image_model
        else None
    )
    text_embedder = (
        SentenceTransformerEmbedder(emb.caption_model, device=emb.device)
        if emb.caption_model
        else None
    )
    return Providers(
        llm=build_llm(cfg, db),
        tts=build_tts(cfg),
        detector=detector,
        asr=asr,
        faces=InsightFaceAnalyzer(cfg.faces.model, device=cfg.faces.device),
        image_embedder=image_embedder,
        text_embedder=text_embedder,
    )


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        providers: Providers | None = None,
        *,
        progress: ProgressFn | None = None,
        on_resolved: ResolvedFn | None = None,
        is_canceled: Callable[[], bool] | None = None,
        db: Database | None = None,
    ) -> None:
        """`db`: share an open database (the job service does); otherwise open and own one."""
        self.cfg = cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._owns_db = db is None
        self.db = db or Database(cfg.data_dir / "offscreen.db")
        self.assets = AssetRepo(self.db)
        self.providers = providers or build_providers(cfg, self.db)
        self._progress = progress
        self._on_resolved = on_resolved
        self._is_canceled = is_canceled
        self._store = ArtifactStore(cfg.data_dir / "artifacts")

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ---- stage graph -------------------------------------------------------------------
    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages(RunOptions())]

    def _stages(self, opts: RunOptions) -> list[Stage]:
        cfg, p = self.cfg, self.providers
        voice = opts.voice or cfg.tts.default_voice
        script = ScriptSettings(
            target_duration_s=max(1, round(opts.minutes * 60)),
            voice_id=voice,
            style=opts.style,
            spoil_ending=opts.spoil_ending,
        )
        models = task_models(
            cfg,
            [
                "story",
                "script_outline",
                "script_write",
                "shot_caption",
                "scene_segment",
                "character_name",
            ],
        )
        caption_task = cfg.tasks.get("shot_caption")
        return [
            ProxyStage(self.assets),
            ShotsStage(self.assets, p.detector),
            KeyframesStage(),
            FacesStage(p.faces),
            CharactersStage(),
            NamingStage(p.llm, models),
            EmbeddingsStage(p.image_embedder, p.text_embedder),
            CaptionsStage(
                p.llm,
                models,
                (caption_task.shots_per_request if caption_task else None) or DEFAULT_BATCH,
            ),
            ScenesStage(p.llm, models),
            TranscriptStage(self.assets, p.asr),
            StoryStage(p.llm, models),
            OutlineStage(
                p.llm,
                OutlineSettings(script.target_duration_s, opts.style, opts.spoil_ending),
                models,
            ),
            ScriptStage(p.llm, script, models),
            PlanStage(p.tts),
            CompileStage(self.assets),
            RenderStage(self.assets),
        ]

    def stage_chain(self, target: str, asset_id: str, opts: RunOptions | None = None) -> list[str]:
        """`target` and everything it needs, upstream first (what `ensure` will walk)."""
        stages = {s.name: s for s in self._stages(opts or RunOptions())}
        if target not in stages:
            raise ValueError(f"unknown stage {target!r}; known: {', '.join(stages)}")
        order: list[str] = []

        def visit(name: str) -> None:
            if name in order:
                return
            for dep in stages[name].inputs({"asset_id": asset_id}):
                visit(dep.stage)
            order.append(name)

        visit(target)
        return order

    def peek(self, stage: str, asset_id: str, opts: RunOptions | None = None) -> Artifact | None:
        """The cached artifact of `stage` if it and all upstream are built; runs nothing."""
        return self._engine(opts or RunOptions(), None).peek(stage, {"asset_id": asset_id})

    def _engine(self, opts: RunOptions, reports: list[StageReport] | None) -> Engine:
        def resolved(stage: str, hit: bool) -> None:
            if reports is not None:
                reports.append(StageReport(stage, hit))
            if self._on_resolved is not None:
                self._on_resolved(stage, hit)

        runs = StageRunRepo(self.db)

        def ran(run: StageRun) -> None:
            runs.add(
                stage=run.stage,
                cache_key=run.cache_key,
                asset_id=run.scope.get("asset_id"),
                job_id=current_job_id(),
                status=run.status,
                error=run.error,
                started_at=run.started_at,
                duration_ms=run.duration_ms,
            )

        return Engine(
            self._store,
            self._stages(opts),
            progress=self._progress,
            is_canceled=self._is_canceled,
            on_resolved=resolved,
            on_run=ran,
        )

    # ---- use cases ---------------------------------------------------------------------
    def import_movie(self, path: Path, title: str | None = None) -> str:
        """Register a movie file (or find it again) and return its asset id."""
        return ingest(path, self.assets, title=title).asset.id

    def resolve_asset(self, asset: str) -> str:
        """`asset` is an asset id, or the path of a movie file to import."""
        if asset.startswith("ast_"):
            if self.assets.get(asset) is None:
                raise ValueError(f"unknown asset {asset}")
            return asset
        return self.import_movie(Path(asset))

    def run_stage(self, stage: str, asset: str, opts: RunOptions | None = None) -> RunResult:
        opts = opts or RunOptions()
        if stage not in self.stage_names():
            raise ValueError(f"unknown stage {stage!r}; known: {', '.join(self.stage_names())}")
        asset_id = self.resolve_asset(asset)
        reports: list[StageReport] = []
        artifact = self._engine(opts, reports).ensure(stage, {"asset_id": asset_id})
        found = self.assets.get(asset_id)
        return RunResult(asset_id, found.title if found else asset_id, artifact, reports)

    def run_all(self, movie: Path, opts: RunOptions | None = None) -> RunResult:
        """Movie file -> commentary video."""
        return self.run_stage(FINAL_STAGE, str(movie), opts)
