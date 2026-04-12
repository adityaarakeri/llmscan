from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from llmscan.cli import app

runner = CliRunner()

_HF_RESPONSE = [
    {"modelId": "TheBloke/Llama-2-7B-GGUF",     "downloads": 1000, "likes": 50, "tags": [], "lastModified": ""},
    {"modelId": "TheBloke/Llama-2-13B-GGUF",    "downloads":  800, "likes": 30, "tags": [], "lastModified": ""},
    {"modelId": "TheBloke/Llama-2-70B-GGUF",    "downloads":  500, "likes": 20, "tags": [], "lastModified": ""},
    {"modelId": "TheBloke/Mistral-7B-GGUF",      "downloads":  900, "likes": 40, "tags": [], "lastModified": ""},
    {"modelId": "someone/NoParamsInName-GGUF",   "downloads":  100, "likes":  5, "tags": [], "lastModified": ""},
]


def _mock_resp(data=_HF_RESPONSE):
    m = MagicMock()
    m.status_code = 200
    m.content = json.dumps(data).encode()
    m.headers = {}
    m.raise_for_status = MagicMock()
    m.json.return_value = data
    return m


class TestMinParamsFilter:
    def test_min_params_excludes_models_below_threshold(self):
        """--min-params 10 removes 7B models from JSON output."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "10", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert not any("7B" in rid for rid in ids)

    def test_min_params_keeps_models_at_or_above_threshold(self):
        """--min-params 13 keeps 13B and 70B models."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "13", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert any("13B" in rid for rid in ids)
        assert any("70B" in rid for rid in ids)

    def test_min_params_zero_shows_all_with_inferable_params(self):
        """--min-params 0 does not filter out any models with inferable param counts."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "0", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # All models with param counts in name should appear
        ids = [r["repo_id"] for r in data]
        assert any("7B" in rid for rid in ids)

    def test_min_params_excludes_models_with_no_inferable_params(self):
        """When --min-params is set, models with no param count in name are excluded."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert not any("NoParams" in rid for rid in ids)


class TestMaxParamsFilter:
    def test_max_params_excludes_models_above_threshold(self):
        """--max-params 10 removes 13B and 70B models."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--max-params", "10", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert not any("13B" in rid for rid in ids)
        assert not any("70B" in rid for rid in ids)

    def test_max_params_keeps_models_at_or_below_threshold(self):
        """--max-params 7 keeps 7B models."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--max-params", "7", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert any("7B" in rid for rid in ids)

    def test_max_params_excludes_models_with_no_inferable_params(self):
        """When --max-params is set, models with no param count in name are excluded."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--max-params", "100", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert not any("NoParams" in rid for rid in ids)


class TestMinMaxParamsCombined:
    def test_min_and_max_params_together_returns_range(self):
        """--min-params 7 --max-params 13 returns only 7B and 13B models."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "7", "--max-params", "13", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert any("7B" in rid for rid in ids)
        assert any("13B" in rid for rid in ids)
        assert not any("70B" in rid for rid in ids)

    def test_impossible_range_returns_empty(self):
        """--min-params 70 --max-params 7 returns no results."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--min-params", "70", "--max-params", "7", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_no_params_filter_includes_models_without_inferable_params(self):
        """Without --min-params/--max-params, models with no param count appear in results."""
        with patch("httpx.get", return_value=_mock_resp()):
            result = runner.invoke(app, ["search", "llama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [r["repo_id"] for r in data]
        assert any("NoParams" in rid for rid in ids)
