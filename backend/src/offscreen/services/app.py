"""The services of one running application (API process, CLI, tests), sharing one database."""

from __future__ import annotations

from offscreen.config import AppConfig
from offscreen.services.annotations import AnnotationService
from offscreen.services.characters import CharacterService
from offscreen.services.events import JobWatcher
from offscreen.services.files import FileService
from offscreen.services.index import IndexService
from offscreen.services.jobs import JobService
from offscreen.services.library import LibraryService
from offscreen.services.pipeline import Providers
from offscreen.services.report import ReportService
from offscreen.services.search import SearchService
from offscreen.store.db import Database


class AppServices:
    def __init__(self, cfg: AppConfig, *, providers: Providers | None = None) -> None:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.db = Database(cfg.data_dir / "offscreen.db")
        self.jobs = JobService(cfg, providers=providers, db=self.db)
        self.library = LibraryService(cfg, self.db, self.jobs)
        self.annotations = AnnotationService(cfg, self.db, self.jobs)
        self.characters = CharacterService(cfg, self.db, self.jobs)
        self.index = IndexService(cfg, self.db, self.jobs)
        self.search = SearchService(cfg, self.db, self.jobs)
        self.report = ReportService(cfg, self.db, self.jobs)
        self.files = FileService(cfg)

    def job_watcher(self) -> JobWatcher:
        """A watcher for one live connection (each keeps its own baseline)."""
        return JobWatcher(self.jobs.jobs)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> AppServices:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
