"""One error shape for every failure: `{"error": {"code", "message", "details"?}}`.

`code` is stable and meant for programs (and the generated frontend types); `message` is for
people. Service errors map to 404 / 422 / 409, request validation to 422, anything unexpected to
a 500 that never leaks internals."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from offscreen.services.errors import Conflict, InvalidInput, NotFound, ServiceError

logger = logging.getLogger(__name__)

_SERVICE_STATUS: dict[type[ServiceError], int] = {NotFound: 404, InvalidInput: 422, Conflict: 409}
_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    416: "range_not_satisfiable",
    422: "invalid_input",
}


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


def error_response(
    status: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Not found"},
    409: {"model": ErrorResponse, "description": "Not allowed in the current state"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
}
"""Documented on every route, so the OpenAPI schema (and the generated client) knows the shape."""


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        status = next((s for t, s in _SERVICE_STATUS.items() if isinstance(exc, t)), 400)
        return error_response(status, exc.code, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": [str(p) for p in e["loc"]], "message": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return error_response(422, "validation_error", "the request is not valid", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return error_response(500, "internal", "internal error")
