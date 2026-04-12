from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.catalog import DEFAULT_MODELS
from llmscan.cli import app

runner = CliRunner()

# A minimal valid model entry not in the bundled catalog
_NEW_MODEL = {
    "id": "new-remote-model-99b",
    "family": "TestFamily",
    "params_b": 99,
    "quant": "Q4_K_M",
    "min_vram_gb": 50.0,
    "recommended_vram_gb": 65.0,
    "recommended_ram_gb": 100.0,
    "notes": "Remote-only test model.",
}

# An existing bundled model with a changed field
_UPDATED_MODEL = dict(DEFAULT_MODELS[0])
_UPDATED_MODEL = {**DEFAULT_MODELS[0], "notes": "Updated note from remote."}

_REMOTE_CATALOG_WITH_NEW = [_NEW_MODEL]
_REMOTE_CATALOG_WITH_UPDATE = [_UPDATED_MODEL]
_REMOTE_CATALOG_MIXED = [_NEW_MODEL, _UPDATED_MODEL]


def _make_httpx_response(data: list, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.content = json.dumps(data).encode()
    mock.headers = {}
    mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    if status >= 400:
        import httpx
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=mock
        )
    return mock


@pytest.fixture(autouse=True)
def _reset_cache():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestCatalogUpdateDryRun:
    def test_dry_run_does_not_save_catalog(self, monkeypatch, tmp_path):
        """--dry-run must not write any file."""
        saved = []
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update", "--dry-run"])

        assert result.exit_code == 0
        assert saved == [], "save_user_catalog must not be called during --dry-run"

    def test_dry_run_shows_new_model_id(self, monkeypatch):
        """--dry-run output includes the ID of a new model from the remote catalog."""
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update", "--dry-run"])

        assert result.exit_code == 0
        assert "new-remote-model-99b" in result.output

    def test_dry_run_shows_dry_run_indicator(self, monkeypatch):
        """--dry-run output makes clear no changes were written."""
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update", "--dry-run"])

        assert result.exit_code == 0
        assert "dry" in result.output.lower() or "preview" in result.output.lower() or "not saved" in result.output.lower()


class TestCatalogUpdateApply:
    def test_apply_saves_new_models_to_user_catalog(self, monkeypatch):
        """Without --dry-run, new remote models are written to the user catalog."""
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert len(saved) == 1
        ids = [m["id"] for m in saved[0]]
        assert "new-remote-model-99b" in ids

    def test_apply_shows_success_message(self, monkeypatch):
        """Apply mode prints a confirmation that changes were saved."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert "saved" in result.output.lower() or "updated" in result.output.lower() or "applied" in result.output.lower()

    def test_apply_preserves_user_custom_models(self, monkeypatch):
        """User-added models not in the remote catalog are kept after update."""
        user_custom = {
            "id": "my-custom-local-model",
            "family": "Custom",
            "params_b": 3,
            "quant": "Q4_K_M",
            "min_vram_gb": 2.0,
            "recommended_vram_gb": 3.0,
            "recommended_ram_gb": 5.0,
            "notes": "Custom user model.",
        }
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [user_custom])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert len(saved) == 1
        ids = [m["id"] for m in saved[0]]
        assert "my-custom-local-model" in ids


class TestCatalogUpdateDiff:
    def test_diff_reports_new_model_count(self, monkeypatch):
        """Output mentions how many new models were found."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert "1" in result.output or "new" in result.output.lower()

    def test_diff_reports_updated_model(self, monkeypatch):
        """Output mentions which model was updated when a bundled model has changed fields."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_UPDATE)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert "updated" in result.output.lower() or DEFAULT_MODELS[0]["id"] in result.output

    def test_no_changes_when_remote_matches_bundled(self, monkeypatch):
        """When remote catalog equals bundled, output says nothing to update."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(list(DEFAULT_MODELS))):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code == 0
        assert (
            "no changes" in result.output.lower()
            or "up to date" in result.output.lower()
            or "nothing" in result.output.lower()
            or "0 new" in result.output.lower()
        )


class TestCatalogUpdateErrors:
    def test_network_error_shows_clear_message(self, monkeypatch):
        """A network failure exits nonzero and prints a user-readable error."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code != 0
        assert "error" in result.output.lower() or "failed" in result.output.lower()

    def test_invalid_json_from_remote_shows_error(self, monkeypatch):
        """Non-JSON response exits nonzero with a clear error."""
        mock = MagicMock()
        mock.content = b"not json at all"
        mock.headers = {}
        mock.raise_for_status = MagicMock()
        mock.json.side_effect = Exception("not valid json")

        with patch("httpx.get", return_value=mock):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "error" in result.output.lower() or "json" in result.output.lower()

    def test_remote_catalog_failing_validation_shows_error(self, monkeypatch):
        """Remote catalog with missing required fields exits nonzero."""
        bad_catalog = [{"id": "bad-model"}]  # missing required fields

        with patch("httpx.get", return_value=_make_httpx_response(bad_catalog)):
            result = runner.invoke(app, ["catalog", "update"])

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "error" in result.output.lower() or "missing" in result.output.lower()

    def test_custom_url_is_used(self, monkeypatch):
        """--url flag overrides the default remote URL."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        called_urls = []

        def mock_get(url, **kwargs):
            called_urls.append(url)
            return _make_httpx_response(list(DEFAULT_MODELS))

        with patch("httpx.get", side_effect=mock_get):
            result = runner.invoke(app, ["catalog", "update", "--url", "https://example.com/catalog.json"])

        assert result.exit_code == 0
        assert called_urls == ["https://example.com/catalog.json"]


class TestCatalogUpdateJson:
    def test_json_output_has_new_and_updated_keys(self, monkeypatch):
        """--json output contains 'new' and 'updated' lists."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_MIXED)):
            result = runner.invoke(app, ["catalog", "update", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "new" in data
        assert "updated" in data
        assert isinstance(data["new"], list)
        assert isinstance(data["updated"], list)

    def test_json_dry_run_does_not_save(self, monkeypatch):
        """--json --dry-run does not call save_user_catalog."""
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_MIXED)):
            result = runner.invoke(app, ["catalog", "update", "--json", "--dry-run"])

        assert result.exit_code == 0
        assert saved == []

    def test_json_new_list_contains_new_model_id(self, monkeypatch):
        """--json 'new' list contains the ID of the new remote model."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        with patch("httpx.get", return_value=_make_httpx_response(_REMOTE_CATALOG_WITH_NEW)):
            result = runner.invoke(app, ["catalog", "update", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "new-remote-model-99b" in data["new"]
