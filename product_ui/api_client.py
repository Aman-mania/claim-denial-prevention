"""Small FastAPI client used by the role-aware Streamlit product UI."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = os.getenv("CLAIM_DENIAL_API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CLAIM_DENIAL_UI_API_TIMEOUT_SECONDS", "45"))


class ApiClientError(RuntimeError):
    """Raised when the product UI cannot complete an API request."""


class ClaimDenialApiClient:
    def __init__(self, *, base_url: str | None = None, token: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.token = token
        self.timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else float(timeout)

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
