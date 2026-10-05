from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.job import Job, JobStatus, Lane
from offscreen.services.errors import NotFound

router = APIRouter(prefix="/jobs", tags=["jobs"], responses=ERROR_RESPONSES)


@router.get("")
def list_jobs(
    services: Services,
    status: Annotated[JobStatus | None, Query()] = None,
    lane: Annotated[Lane | None, Query()] = None,
) -> list[Job]:
    """Newest first."""
    return services.jobs.list(status=status, lane=lane)


@router.get("/{job_id}")
def get_job(job_id: str, services: Services) -> Job:
    job = services.jobs.get(job_id)
    if job is None:
        raise NotFound(f"unknown job {job_id}")
    return job


@router.get(
    "/{job_id}/log",
    responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}}}},
)
def get_job_log(job_id: str, services: Services) -> PlainTextResponse:
    """The end of the job's log file, as plain text."""
    return PlainTextResponse(services.jobs.log(job_id))


@router.post("/{job_id}:cancel")
def cancel_job(job_id: str, services: Services) -> Job:
    """Queued jobs are canceled at once; a running job stops at its next checkpoint."""
    return services.jobs.cancel(job_id)


@router.post("/{job_id}:retry")
def retry_job(job_id: str, services: Services) -> Job:
    """Only failed jobs can be retried."""
    return services.jobs.retry(job_id)
