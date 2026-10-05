from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from offscreen.services.app import AppServices


def get_services(request: Request) -> AppServices:
    return cast(AppServices, request.app.state.services)


Services = Annotated[AppServices, Depends(get_services)]
