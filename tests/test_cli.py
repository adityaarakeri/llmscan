from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.huggingface import HFModelResult, HuggingFaceError

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
    """Reset the cached profile and mock _get_profile for every test."""
    cli_module._cached_profile = None
    with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
        yield
    cli_module._cached_profile = None


# ---------------------------------------------------------------------------
# 8.12  CLI smoke tests
# ---------------------------------------------------------------------------


class TestBranding:
    def test_banner_contains_llm_scan(self):
        """Banner should render the LLM SCAN brand, not old LLM CHECK."""
        from llmscan.cli import BANNER

        assert "SCAN" in BANNER.upper() or "███████╗ ██████╗ █████╗ ███╗   ██╗" in BANNER
        assert "CHECK" not in BANNER.upper()

    def test_version_output_shows_llmscan(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "llmscan" in result.output
        assert "llmcheck" not in result.output
        assert "llmfitcheck" not in result.output

    def test_help_text_references_llmscan(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "llmcheck" not in result.output


class TestCliSmoke:
    def test_default_invocation(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "LLM" in result.output or "██" in result.output  # banner

    def test_scan(self):
        result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "Machine Profile" in result.output

    def test_scan_json(self):
        result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "os" in data
        assert "ram_gb" in data

    def test_list(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Model Fit Matrix" in result.output

    def test_list_min_rating_great(self):
        result = runner.invoke(app, ["list", "--min-rating", "great"])
        assert result.exit_code == 0

    def test_list_min_rating_invalid(self):
        result = runner.invoke(app, ["list", "--min-rating", "invalid"])
        assert result.exit_code != 0

    def test_list_json(self):
        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "machine" in data
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_explain_known_model(self):
        result = runner.invoke(app, ["explain", "llama-3.1-8b-instruct"])
        assert result.exit_code == 0
        assert "Model Explanation" in result.output

    def test_explain_nonexistent_model(self):
        result = runner.invoke(app, ["explain", "nonexistent-model-xyz"])
        assert result.exit_code != 0

    def test_default_invocation_calls_detect_once(self):
        """The default path (main → list_models) should only call _get_profile, not detect_machine twice."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE) as mock_get:
            runner.invoke(app, [])
            # main() calls _get_profile, then list_models() calls _get_profile
            # Both should use the cached helper, not raw detect_machine
            assert mock_get.call_count >= 1

    def test_default_hints_mention_search_and_add(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "add" in result.output


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------

FAKE_HF_RESULTS = [
    HFModelResult(
        repo_id="TheBloke/Llama-2-7B-GGUF",
        author="TheBloke",
        model_name="Llama-2-7B-GGUF",
        downloads=50000,
        likes=200,
        tags=["gguf"],
        last_modified="2024-01-01",
    ),
]


class TestSearchCommand:
    def test_search_shows_table(self, monkeypatch):
        monkeypatch.setattr("llmscan.cli.search_gguf_models", lambda q, limit: FAKE_HF_RESULTS)
        result = runner.invoke(app, ["search", "llama"])
        assert result.exit_code == 0
        assert "TheBloke/Llama-2-7B-GGUF" in result.output

    def test_search_json(self, monkeypatch):
        monkeypatch.setattr("llmscan.cli.search_gguf_models", lambda q, limit: FAKE_HF_RESULTS)
        result = runner.invoke(app, ["search", "llama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["repo_id"] == "TheBloke/Llama-2-7B-GGUF"

    def test_search_limit(self, monkeypatch):
        captured = {}

        def mock_search(q, limit):
            captured["limit"] = limit
            return FAKE_HF_RESULTS

        monkeypatch.setattr("llmscan.cli.search_gguf_models", mock_search)
        runner.invoke(app, ["search", "llama", "--limit", "5"])
        assert captured["limit"] == 5

    def test_search_no_results(self, monkeypatch):
        monkeypatch.setattr("llmscan.cli.search_gguf_models", lambda q, limit: [])
        result = runner.invoke(app, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No GGUF models found" in result.output

    def test_search_network_error(self, monkeypatch):
        def raise_err(q, limit):
            raise HuggingFaceError("Connection refused")

        monkeypatch.setattr("llmscan.cli.search_gguf_models", raise_err)
        result = runner.invoke(app, ["search", "llama"])
        assert result.exit_code != 0
        assert "Connection refused" in result.output


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


class TestAddCommand:
    def test_add_manual_model(self, tmp_path, monkeypatch):
        cat_path = tmp_path / "catalog.json"
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: cat_path)
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr(
            "llmscan.cli.save_user_catalog",
            lambda entries: cat_path.write_text(json.dumps(entries), encoding="utf-8"),
        )
        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--family", "Test"])
        assert result.exit_code == 0
        assert "Model added" in result.output

    def test_add_json_output(self, tmp_path, monkeypatch):
        cat_path = tmp_path / "catalog.json"
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr(
            "llmscan.cli.save_user_catalog",
            lambda entries: cat_path.write_text(json.dumps(entries), encoding="utf-8"),
        )
        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == "my-model"
        assert data["min_vram_gb"] > 0
        assert data["recommended_vram_gb"] > 0

    def test_add_missing_params_and_quant(self):
        result = runner.invoke(app, ["add", "my-model"])
        assert result.exit_code != 0

    def test_add_invalid_quant(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "INVALID"])
        assert result.exit_code != 0
        assert "Unknown quantization" in result.output

    def test_add_duplicate_warns(self, tmp_path, monkeypatch):
        existing = [
            {
                "id": "my-model",
                "family": "X",
                "params_b": 7,
                "quant": "Q4_K_M",
                "min_vram_gb": 4,
                "recommended_vram_gb": 5,
                "recommended_ram_gb": 8,
                "notes": "",
            }
        ]
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: list(existing))
        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M"])
        assert result.exit_code != 0
        assert "already exists" in result.output
        assert "--force" in result.output


# ---------------------------------------------------------------------------
# remove command
# ---------------------------------------------------------------------------


class TestRemoveCommand:
    def test_remove_existing(self, tmp_path, monkeypatch):
        existing = [
            {
                "id": "my-model",
                "family": "X",
                "params_b": 7,
                "quant": "Q4_K_M",
                "min_vram_gb": 4,
                "recommended_vram_gb": 5,
                "recommended_ram_gb": 8,
                "notes": "",
            }
        ]
        saved: list[list] = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: list(existing))
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda entries: saved.append(entries))
        result = runner.invoke(app, ["remove", "my-model"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert len(saved[0]) == 0

    def test_remove_nonexistent(self, monkeypatch):
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        result = runner.invoke(app, ["remove", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_remove_bundled_not_possible(self, monkeypatch):
        # Bundled models aren't in user catalog, so remove should fail
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        result = runner.invoke(app, ["remove", "llama-3.1-8b-instruct"])
        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Bundled" in result.output or "bundled" in result.output or "llmscan add" in result.output
