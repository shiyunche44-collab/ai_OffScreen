"""The composition root: the API server and the job worker in one process (ADR-0003).

uvicorn runs on the calling thread and serves the API (plus the built front end, if there is
one); the worker runs on a background thread and executes the jobs the API queues. Stopping the
process stops the worker cooperatively and puts jobs it did not finish back in the queue, so the
next start resumes them. `run_worker` is the worker alone, for a second process or for development
with `serve --no-worker`."""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

import uvicorn

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.log import setup_logging
from offscreen.services.app import AppServices
from offscreen.services.pipeline import Providers
from offscreen.worker import Worker, WorkerSettings

logger = logging.getLogger(__name__)

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
SHUTDOWN_GRACE_S = 10.0


def worker_settings(cfg: AppConfig) -> WorkerSettings:
    return WorkerSettings(
        lane_limits={
            "gpu": 1,
            "cpu": cfg.worker.cpu_concurrency,
            "api": cfg.worker.api_concurrency,
        },
        heartbeat_timeout_s=cfg.worker.heartbeat_timeout_s,
    )


def build_worker(services: AppServices) -> Worker:
    return Worker(
        services.jobs.jobs,
        services.jobs.execute,
        services.cfg.data_dir,
        worker_settings(services.cfg),
    )


def default_web_dir() -> Path | None:
    """`frontend/dist` of this checkout, if it has been built."""
    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    return dist if (dist / "index.html").is_file() else None


def _logging(cfg: AppConfig) -> None:
    setup_logging(
        cfg.data_dir / "logs", secret_env_names=[p.api_key_env for p in cfg.providers.values()]
    )


class Server:
    """API (+ front end) and, unless disabled, the worker. `run()` blocks until stopped."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        web_dir: Path | None = None,
        with_worker: bool = True,
        providers: Providers | None = None,
    ) -> None:
        self.cfg = cfg
        self.host = host
        self.web_dir = web_dir
        self.with_worker = with_worker
        self.services = AppServices(cfg, providers=providers)
        self.worker = build_worker(self.services) if with_worker else None
        app = create_app(self.services, web_dir=web_dir)
        self._uvicorn = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="info", log_config=None)
        )
        self.started = threading.Event()

    @property
    def port(self) -> int:
        """The port actually bound (differs from the requested one when that was 0)."""
        sockets = self._uvicorn.servers[0].sockets if self._uvicorn.servers else []
        return int(sockets[0].getsockname()[1]) if sockets else self._uvicorn.config.port

    def request_stop(self) -> None:
        self._uvicorn.should_exit = True

    def run(self) -> None:
        if self.host not in LOOPBACK:
            logger.warning(
                "listening on %s: the server has no authentication, anyone who can reach it "
                "can use your model keys and read your movies",
                self.host,
            )
        thread = None
        if self.worker is not None:
            thread = threading.Thread(target=self.worker.run_forever, name="worker", daemon=True)
            thread.start()
        watcher = threading.Thread(target=self._announce, daemon=True)
        watcher.start()
        try:
            self._uvicorn.run()
        finally:
            if self.worker is not None:
                self.worker.stop()
                self.worker.shutdown(SHUTDOWN_GRACE_S)
            if thread is not None:
                thread.join(SHUTDOWN_GRACE_S)
            self.services.close()

    def _announce(self) -> None:
        while not self._uvicorn.started and not self._uvicorn.should_exit:
            time.sleep(0.02)
        if self._uvicorn.started:
            self.started.set()
            where = "api + worker" if self.with_worker else "api only"
            front = "with front end" if self.web_dir else "without front end"
            logger.info("serving %s %s on http://%s:%d", where, front, self.host, self.port)


def serve(
    cfg: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    web_dir: Path | None = None,
    with_worker: bool = True,
) -> None:
    """What `offscreen serve` runs."""
    _logging(cfg)
    Server(cfg, host=host, port=port, web_dir=web_dir, with_worker=with_worker).run()


def run_worker(
    cfg: AppConfig,
    *,
    providers: Providers | None = None,
    stop: threading.Event | None = None,
    install_signals: bool = True,
) -> None:
    """The worker alone (`offscreen worker`): runs queued jobs until stopped."""
    _logging(cfg)
    with AppServices(cfg, providers=providers) as services:
        worker = build_worker(services)
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: worker.stop())
        if stop is not None:

            def watch() -> None:
                stop.wait()
                worker.stop()

            threading.Thread(target=watch, daemon=True, name="stop-watch").start()
        worker.run_forever()


__all__ = ["Server", "default_web_dir", "run_worker", "serve", "worker_settings"]
