"""The FastAPI application: a thin layer over `services` (no media work, no model calls)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from offscreen.api.errors import install_error_handlers
from offscreen.api.routers import assets, events, files, jobs, projects, styles
from offscreen.api.sse import EventSettings
from offscreen.api.static import mount_frontend
from offscreen.config import load_config
from offscreen.services.app import AppServices

API_PREFIX = "/api"
TITLE = "AI OffScreen"
VERSION = "0.1.0"


def create_app(
    services: AppServices | None = None,
    *,
    events_settings: EventSettings | None = None,
    web_dir: Path | None = None,
) -> FastAPI:
    """`services`: ready-made services (tests). Without them the app loads the config
    (`$OFFSCREEN_CONFIG` or ./config.yaml) when it starts and closes the services when it stops.
    `web_dir`: the built front end to serve next to the API (production mode)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        own = AppServices(load_config())
        app.state.services = own
        try:
            yield
        finally:
            own.close()

    app = FastAPI(title=TITLE, version=VERSION, lifespan=lifespan)
    app.state.event_settings = events_settings or EventSettings()
    install_error_handlers(app)
    for module in (assets, projects, styles, jobs, files, events):
        app.include_router(module.router, prefix=API_PREFIX)
    if web_dir is not None:
        mount_frontend(app, web_dir)  # last: its catch-all must not shadow the API
    return app
