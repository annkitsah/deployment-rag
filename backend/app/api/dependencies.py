from fastapi import Request

from app.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    """Return the application's dependency container from app state."""

    return request.app.state.container