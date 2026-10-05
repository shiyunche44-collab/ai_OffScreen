from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES, error_response
from offscreen.services.files import check_range

router = APIRouter(tags=["files"], responses=ERROR_RESPONSES)

IMMUTABLE = "public, max-age=31536000, immutable"


@router.head("/files/{path:path}", include_in_schema=False)  # same handler, not a second operation
@router.get(
    "/files/{path:path}",
    response_class=FileResponse,
    responses={
        200: {"description": "The file", "content": {"application/octet-stream": {}}},
        206: {"description": "The requested byte range(s)"},
        416: {"description": "Range not satisfiable"},
    },
)
def get_file(path: str, request: Request, services: Services) -> Response:
    """A file under the data directory (videos, keyframes, audio, ...). Supports HTTP Range, so
    players can seek. Nothing outside the data directory is reachable."""
    file = services.files.resolve(path)
    size = file.stat().st_size
    if (wanted := request.headers.get("range")) is not None:
        verdict = check_range(wanted, size)
        if verdict == "malformed":
            return error_response(400, "bad_request", "malformed Range header")
        if verdict == "unsatisfiable":
            refusal = error_response(416, "range_not_satisfiable", f"the file has {size} bytes")
            refusal.headers["Content-Range"] = f"bytes */{size}"
            return refusal
    response = FileResponse(file)
    if services.files.is_immutable(path):
        response.headers["Cache-Control"] = IMMUTABLE
    return response
