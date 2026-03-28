"""Security tests covering findings F1–F12 from PLAN.md Section 11."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.catalog import _MAX_CATALOG_SIZE_BYTES, load_user_catalog
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.estimator import load_catalog
from llmscan.huggingface import (
    HuggingFaceError,
    _sanitize_error,
    get_model_files,
    search_gguf_models,
    validate_repo_id,
)

runner = CliRunner()

FAKE_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Test CPU",
    ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)


@pytest.fixture(autouse=True)
def _reset_cache_and_mock():
    cli_module._cached_profile = None
    with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
        yield
    cli_module._cached_profile = None


# ---------------------------------------------------------------------------
# F1 — Rich markup injection
# ---------------------------------------------------------------------------


class TestRichMarkupInjection:
    def test_add_with_markup_family_is_escaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(
            app,
            ["add", "test-safe-model", "--params-b", "7", "--quant", "Q4_K_M", "--family", "[red]INJECT[/red]"],
        )
        assert result.exit_code == 0
        # The literal brackets should appear escaped, not interpreted as Rich tags
        assert "\\[red]" in result.output or "[red]INJECT" in result.output

    def test_explain_with_markup_notes_is_escaped(self, tmp_path, monkeypatch):
        catalog = [
            {
                "id": "markup-test",
                "family": "Test",
                "params_b": 7,
                "quant": "Q4_K_M",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "[link=http://evil.com]click here[/link]",
            }
        ]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        result = runner.invoke(app, ["explain", "markup-test", "--catalog", str(catalog_path)])
        assert result.exit_code == 0
        # Should not contain a clickable link — the markup should be escaped
        assert "\\[link=" in result.output or "[link=" in result.output

    def test_list_with_markup_id_is_escaped(self, tmp_path):
        catalog = [
            {
                "id": "[bold]fake[/bold]",
                "family": "Test",
                "params_b": 7,
                "quant": "Q4_K_M",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "test",
            }
        ]
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        result = runner.invoke(app, ["list", "--catalog", str(catalog_path), "--min-rating", "no"])
        assert result.exit_code == 0

    def test_remove_markup_model_id_is_escaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(app, ["remove", "[red]evil[/red]"])
        # Should fail (not found), but should not render red text
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# F2 — HTTP response size limits
# ---------------------------------------------------------------------------


class TestResponseSizeLimits:
    def test_search_rejects_oversized_response(self):
        large_body = b"x" * (6 * 1024 * 1024)  # 6 MB
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(large_body))}
        mock_response.raise_for_status = MagicMock()
        mock_response.content = large_body

        with (
            patch("llmscan.huggingface.httpx.get", return_value=mock_response),
            pytest.raises(HuggingFaceError, match="too large"),
        ):
            search_gguf_models("llama")

    def test_get_model_files_rejects_oversized_response(self):
        large_body = b"x" * (6 * 1024 * 1024)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-length": str(len(large_body))}
        mock_response.raise_for_status = MagicMock()
        mock_response.content = large_body

        with (
            patch("llmscan.huggingface.httpx.get", return_value=mock_response),
            pytest.raises(HuggingFaceError, match="too large"),
        ):
            get_model_files("owner/repo")


# ---------------------------------------------------------------------------
# F3 — repo_id validation
# ---------------------------------------------------------------------------


class TestRepoIdValidation:
    def test_valid_repo_id(self):
        validate_repo_id("TheBloke/Llama-2-7B-GGUF")  # Should not raise

    def test_valid_repo_id_with_dots(self):
        validate_repo_id("user.name/model.name")  # Should not raise

    def test_traversal_rejected(self):
        with pytest.raises(HuggingFaceError, match="Invalid repo ID"):
            validate_repo_id("attacker/../admin")

    def test_query_params_rejected(self):
        with pytest.raises(HuggingFaceError, match="Invalid repo ID"):
            validate_repo_id("attacker/model?q=evil")

    def test_empty_rejected(self):
        with pytest.raises(HuggingFaceError, match="Invalid repo ID"):
            validate_repo_id("")

    def test_no_slash_rejected(self):
        with pytest.raises(HuggingFaceError, match="Invalid repo ID"):
            validate_repo_id("just-a-name")

    def test_get_model_files_validates_repo_id(self):
        with pytest.raises(HuggingFaceError, match="Invalid repo ID"):
            get_model_files("../../../etc/passwd")


# ---------------------------------------------------------------------------
# F4 — catalog path traversal
# ---------------------------------------------------------------------------


class TestCatalogPathTraversal:
    def test_etc_path_rejected(self):
        with pytest.raises(SystemExit, match="not allowed"):
            load_catalog("/etc/evil.json")

    def test_proc_path_rejected(self):
        with pytest.raises(SystemExit, match="not allowed"):
            load_catalog("/proc/self/environ.json")

    def test_sys_path_rejected(self):
        with pytest.raises(SystemExit, match="not allowed"):
            load_catalog("/sys/class/net.json")

    def test_non_json_extension_rejected(self):
        with pytest.raises(SystemExit, match=".json extension"):
            load_catalog("/tmp/catalog.txt")

    def test_valid_json_in_cwd_accepted(self, tmp_path):
        catalog = [
            {
                "id": "test",
                "family": "Test",
                "params_b": 7,
                "quant": "Q4_K_M",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "",
            }
        ]
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(catalog))
        result = load_catalog(str(path))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# F5 — HF_TOKEN leakage
# ---------------------------------------------------------------------------


class TestTokenLeakage:
    def test_sanitize_error_redacts_bearer_token(self):
        exc = Exception("Connection failed, headers: Bearer hf_abc123secret")
        sanitized = _sanitize_error(exc)
        assert "hf_abc123secret" not in sanitized
        assert "Bearer ***" in sanitized

    def test_sanitize_error_redacts_env_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_mysecrettoken")
        exc = Exception("Error with hf_mysecrettoken in message")
        sanitized = _sanitize_error(exc)
        assert "hf_mysecrettoken" not in sanitized
        assert "***" in sanitized

    def test_sanitize_error_no_token_unchanged(self):
        exc = Exception("Normal error message")
        sanitized = _sanitize_error(exc)
        assert sanitized == "Normal error message"

    def test_search_error_does_not_leak_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secretvalue")

        def _raise(*args, **kwargs):
            raise httpx.ConnectError("Failed with Bearer hf_secretvalue")

        with patch("llmscan.huggingface.httpx.get", side_effect=_raise):
            with pytest.raises(HuggingFaceError) as exc_info:
                search_gguf_models("test")
            assert "hf_secretvalue" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# F6 — User catalog size check
# ---------------------------------------------------------------------------


class TestUserCatalogSizeCheck:
    def test_oversized_user_catalog_rejected(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        # Write a file just over the limit
        path.write_text("x" * (_MAX_CATALOG_SIZE_BYTES + 1))
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        with pytest.raises(SystemExit, match="too large"):
            load_user_catalog()

    def test_normal_user_catalog_accepted(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([]))
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        result = load_user_catalog()
        assert result == []


# ---------------------------------------------------------------------------
# F7 — Dependency pinning
# ---------------------------------------------------------------------------


class TestDependencyPinning:
    def test_all_dependencies_have_upper_bound(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        for dep in data["project"]["dependencies"]:
            assert "<" in dep, f"Dependency '{dep}' missing upper bound"


# ---------------------------------------------------------------------------
# F8 — Unbounded --limit
# ---------------------------------------------------------------------------


class TestSearchLimitBounds:
    def test_limit_zero_rejected(self):
        result = runner.invoke(app, ["search", "llama", "--limit", "0"])
        assert result.exit_code != 0

    def test_limit_101_rejected(self):
        result = runner.invoke(app, ["search", "llama", "--limit", "101"])
        assert result.exit_code != 0

    @patch("llmscan.cli.search_gguf_models", return_value=[])
    def test_limit_50_accepted(self, mock_search):
        result = runner.invoke(app, ["search", "llama", "--limit", "50"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# F9 — model_id validation
# ---------------------------------------------------------------------------


class TestModelIdValidation:
    def test_valid_model_id_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(
            app,
            ["add", "valid-model-7b", "--params-b", "7", "--quant", "Q4_K_M"],
        )
        assert result.exit_code == 0

    def test_markup_model_id_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(
            app,
            ["add", "[red]evil[/red]", "--params-b", "7", "--quant", "Q4_K_M"],
        )
        assert result.exit_code != 0

    def test_traversal_model_id_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(
            app,
            ["add", "../traversal", "--params-b", "7", "--quant", "Q4_K_M"],
        )
        assert result.exit_code != 0

    def test_empty_model_id_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "catalog.json")
        result = runner.invoke(
            app,
            ["add", "", "--params-b", "7", "--quant", "Q4_K_M"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# F10 — GitHub Actions SHA pinning
# ---------------------------------------------------------------------------


class TestGitHubActionsPinning:
    def test_actions_use_sha_pins(self):
        ci_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        import re

        uses_lines = re.findall(r"uses:\s*(.+)", content)
        for ref in uses_lines:
            ref = ref.strip()
            # Must contain a SHA (40 hex chars)
            assert re.search(r"@[0-9a-f]{40}", ref), f"Action '{ref}' not pinned to SHA"


# ---------------------------------------------------------------------------
# F12 — HF API response type validation
# ---------------------------------------------------------------------------


class TestHFResponseValidation:
    def test_search_rejects_non_list_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"error": "not a list"}'
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "not a list"}

        with (
            patch("llmscan.huggingface.httpx.get", return_value=mock_response),
            pytest.raises(HuggingFaceError, match="Unexpected response"),
        ):
            search_gguf_models("test")

    def test_get_model_files_rejects_non_dict_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'["not a dict"]'
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = ["not a dict"]

        with (
            patch("llmscan.huggingface.httpx.get", return_value=mock_response),
            pytest.raises(HuggingFaceError, match="Unexpected response"),
        ):
            get_model_files("owner/repo")
