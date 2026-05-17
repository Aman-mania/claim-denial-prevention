from __future__ import annotations

import importlib


def test_api_client_exports_backward_compatible_defaults(monkeypatch):
    monkeypatch.setenv("APP_ENV", "docker")
    monkeypatch.setenv("CLAIM_DENIAL_API_BASE_URL", "http://api:8000")
    import product_ui.api_client as client
    importlib.reload(client)
    assert client.DEFAULT_API_BASE_URL == "http://api:8000"
    assert isinstance(client.DEFAULT_API_TIMEOUT_SECONDS, float)
    assert client.ClaimDenialApiClient().base_url == "http://api:8000"
