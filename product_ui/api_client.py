"""Small FastAPI client used by the role-aware Streamlit product UI."""

from __future__ import annotations

from typing import Any

import requests

try:
    from src.config.runtime import get_runtime_settings
except Exception:  # pragma: no cover - defensive fallback for unusual import paths
    get_runtime_settings = None  # type: ignore[assignment]


def _default_base_url() -> str:
    if get_runtime_settings is None:
        return "http://localhost:8000"
    return get_runtime_settings().ui_api_base_url


def _default_timeout() -> float:
    if get_runtime_settings is None:
        return 45.0
    return get_runtime_settings().ui_api_timeout_seconds


# Backward-compatible constants used by product_ui/app.py and older tests.
# They are evaluated from the centralized runtime config so Docker/AWS can still
# switch behavior by environment variables instead of scattered if/else logic.
DEFAULT_API_BASE_URL = _default_base_url()
DEFAULT_API_TIMEOUT_SECONDS = _default_timeout()


class ApiClientError(RuntimeError):
    """Raised when the product UI cannot complete an API request."""


class ClaimDenialApiClient:
    def __init__(self, *, base_url: str | None = None, token: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.token = token
        self.timeout = _default_timeout() if timeout is None else float(timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                headers=self._headers(),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise ApiClientError(f"Could not reach API at {url}: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text}

        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            raise ApiClientError(f"API request failed ({response.status_code}): {detail}")
        if not isinstance(payload, dict):
            raise ApiClientError("API returned a non-object response.")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def artifact_health(self) -> dict[str, Any]:
        return self._request("GET", "/health/artifacts")

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/login", json={"email": email, "password": password})

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me")

    def validate_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/claims/validate", json=claim)

    def recommend_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/claims/recommend", json=claim)
