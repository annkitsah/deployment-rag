"""Simple shared-password gate for the public demo."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config.settings import get_settings


def require_app_password(
    x_app_password: str | None = Header(default=None, alias="X-App-Password"),
) -> None:
    expected = (get_settings().app_password or "").strip()
    if not expected:
        return

    provided = (x_app_password or "").strip()
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing app password.",
            headers={"WWW-Authenticate": "AppPassword"},
        )