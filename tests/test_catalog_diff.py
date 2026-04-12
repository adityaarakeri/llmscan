from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.catalog import DEFAULT_MODELS
from llmscan.cli import app

runner = CliRunner()

_NEW_MODEL = {
    "id": "new-remote-diff-99b",
    "family": "Test",
    "params_b": 99,
    "quant": "Q4_K_M",
    "min_vram_gb": 50.0,
    "recommended_vram_gb": 65.0,
    "recommended_ram_gb": 100.0,
    "notes": "",
}
_UPDATED_MODEL = {**DEFAULT_MODELS[0], "notes": "Updated in remote."}


def _mock_resp(data):
    m = MagicMock()
    m.content = json.dumps(data).encode()
    m.headers = {}
    m.raise_for_status = MagicMock()
    m.json.return_value = data
    return m


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestCatalogDiffRemoved:
    def test_removed_models_shown_when_remote_drops_bundled_entry(self, monkeypatch):
        """When the remote catalog omits a bundled model, it appears as 'removed'."""
        # Remote only has one model, omitting the rest of DEFAULT_MODELS
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_mock_resp([DEFAULT_MODELS[0]])):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower()

    def test_removed_count_matches_missing_bundled_models(self, monkeypatch):
        """The number of removed models equals bundled count minus remote count when IDs differ."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        # Remote only has _NEW_MODEL — none of the bundled models
        with patch("httpx.get", return_value=_mock_resp([_NEW_MODEL])):
            result = runner.invoke(app, ["catalog", "update", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "removed" in data
        assert len(data["removed"]) == len(DEFAULT_MODELS)

    def test_no_removed_when_remote_is_superset(self, monkeypatch):
        """When remote has all bundled models plus extras, removed list is empty."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        remote = list(DEFAULT_MODELS) + [_NEW_MODEL]
        with patch("httpx.get", return_value=_mock_resp(remote)):
            result = runner.invoke(app, ["catalog", "update", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["removed"] == []

    def test_removed_shown_in_diff_table(self, monkeypatch):
        """Diff table includes removed model IDs when remote drops bundled entries."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        # Remote only has one bundled model
        with patch("httpx.get", return_value=_mock_resp([DEFAULT_MODELS[0]])):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        # At least one of the other bundled model IDs should appear in the output
        other_ids = [m["id"] for m in DEFAULT_MODELS[1:3]]
        assert any(mid in result.output for mid in other_ids)

    def test_dry_run_shows_removed_without_saving(self, monkeypatch):
        """--dry-run shows removed models but does not call save."""
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        with patch("httpx.get", return_value=_mock_resp([DEFAULT_MODELS[0]])):
            result = runner.invoke(app, ["catalog", "update", "--dry-run"])

        assert result.exit_code == 0
        assert saved == []
        assert "removed" in result.output.lower()

    def test_json_output_has_removed_key(self, monkeypatch):
        """--json output always includes a 'removed' key."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        remote = list(DEFAULT_MODELS) + [_NEW_MODEL]
        with patch("httpx.get", return_value=_mock_resp(remote)):
            result = runner.invoke(app, ["catalog", "update", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "removed" in data
        assert isinstance(data["removed"], list)
