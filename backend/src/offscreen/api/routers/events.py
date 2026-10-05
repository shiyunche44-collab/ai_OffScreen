from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from offscreen.api.deps import Services
from offscreen.api.sse import EventSettings, job_events

router = APIRouter(tags=["events"])


@router.get(
    "/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "A Server-Sent Events stream. First a `snapshot` event whose data is "
                '`{"jobs": [Job, ...]}` (active jobs and those finished in the last minutes), '
                "then a `job` event carrying the full `Job` whenever one changes. The server "
                "closes the stream now and then; `EventSource` reconnects by itself and gets a "
                "fresh snapshot, so clients just replace jobs by id."
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
def events(request: Request, services: Services) -> StreamingResponse:
    """Live job progress."""
    settings: EventSettings = request.app.state.event_settings
    stream = job_events(services.job_watcher(), request.is_disconnected, settings)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
