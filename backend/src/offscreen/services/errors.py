"""Failures the use cases report to their callers (the API maps them to HTTP statuses)."""

from __future__ import annotations


class ServiceError(Exception):
    """Base class; `code` is the stable machine-readable name."""

    code = "error"


class NotFound(ServiceError):
    code = "not_found"


class InvalidInput(ServiceError, ValueError):
    code = "invalid_input"


class Conflict(ServiceError):
    """The request is fine but the current state does not allow it."""

    code = "conflict"
