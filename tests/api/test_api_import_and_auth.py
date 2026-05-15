from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")


def test_api_app_imports():
    module = importlib.import_module("api.main")
    assert module.app.title == "Claim Denial Prevention API"
