"""Serving the built front end (`frontend/dist`) from the same origin as the API.

Everything under `/api` belongs to the API (ADR-0002); every other path is a file of the build or,
failing that, `index.html` so the browser-side router can handle it."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException

IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"


def mount_frontend(app: FastAPI, dist: Path) -> None:
    """Add the catch-all route. Must be called after the API routers are included."""
    root = dist.resolve()
    index = root / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"{index} not found; build the front end first (npm run build)")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str, request: Request) -> Response:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "no such API route")
        candidate = _inside(root, path) if path else None
        if candidate is not None:
            response = FileResponse(candidate)
            # Vite names built assets after their content, so those never change.
            response.headers["Cache-Control"] = (
                IMMUTABLE if path.startswith("assets/") else NO_CACHE
            )
            return response
        response = FileResponse(index)
        response.headers["Cache-Control"] = NO_CACHE
        return response


def _inside(root: Path, path: str) -> Path | None:
    """The file `path` names inside `root`, or None (missing, a directory, outside, or not a
    valid path at all, e.g. a NUL byte)."""
    try:
        candidate = (root / path).resolve()
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() and candidate.is_relative_to(root) else None
