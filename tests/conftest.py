"""Pytest bootstrap for plugin-local tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def isolated_json_storage(tmp_path, monkeypatch):
    """Route plugin JSON storage to a per-test temporary directory."""

    import src.app.plugin_system.api.storage_api as storage_api
    from src.kernel.storage import JSONStore

    store = JSONStore(str(tmp_path / "json"))
    monkeypatch.setattr(storage_api, "_get_plugin_json_store", lambda _name: store)
    return store
