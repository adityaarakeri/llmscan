from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.huggingface import HFFileInfo

runner = CliRunner()

FAKE_PROFILE = MachineProfile(
    os="Linux", arch="x86_64", cpu="Test CPU", ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestAddDryRunDoesNotSave:
    def test_dry_run_does_not_call_save_user_catalog(self, monkeypatch):
        """--dry-run must not write to the user catalog."""
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert saved == [], "save_user_catalog must not be called with --dry-run"

    def test_dry_run_exits_zero_for_valid_model(self, monkeypatch):
        """--dry-run exits 0 when params and quant are valid."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0

    def test_dry_run_with_hf_repo_does_not_save(self, monkeypatch):
        """--dry-run with HF repo auto-detection does not write to user catalog."""
        saved = []
        monkeypatch.setattr(
            "llmscan.cli.get_model_files",
            lambda r: [HFFileInfo(filename="model.Q4_K_M.gguf", size_bytes=1000)],
        )
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        result = runner.invoke(app, ["add", "TheBloke/Llama-2-7B-GGUF", "--dry-run"])

        assert result.exit_code == 0
        assert saved == []


class TestAddDryRunOutput:
    def test_dry_run_shows_computed_min_vram(self, monkeypatch):
        """--dry-run output includes the computed Min VRAM value."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert "Min VRAM" in result.output or "min_vram" in result.output

    def test_dry_run_shows_computed_recommended_vram(self, monkeypatch):
        """--dry-run output includes the computed Recommended VRAM value."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert "Recommended VRAM" in result.output or "recommended_vram" in result.output

    def test_dry_run_shows_model_id(self, monkeypatch):
        """--dry-run output shows the model ID that would be written."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert "my-model" in result.output

    def test_dry_run_shows_dry_run_indicator(self, monkeypatch):
        """--dry-run output clearly says it is a preview / dry run."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert (
            "dry" in result.output.lower()
            or "preview" in result.output.lower()
            or "not saved" in result.output.lower()
            or "would" in result.output.lower()
        )

    def test_dry_run_does_not_say_model_added(self, monkeypatch):
        """--dry-run output must not claim the model was added/saved."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert "added to local catalog" not in result.output.lower()


class TestAddDryRunJson:
    def test_dry_run_json_outputs_valid_json(self, monkeypatch):
        """--dry-run --json emits valid JSON of the entry that would be written."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(
            app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--json", "--dry-run"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == "my-model"

    def test_dry_run_json_does_not_save(self, monkeypatch):
        """--dry-run --json does not write to user catalog."""
        saved = []
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: saved.append(e))

        runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--json", "--dry-run"])

        assert saved == []

    def test_dry_run_json_has_vram_fields(self, monkeypatch):
        """--dry-run --json output includes min_vram_gb and recommended_vram_gb."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(
            app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--json", "--dry-run"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["min_vram_gb"] > 0
        assert data["recommended_vram_gb"] > 0
        assert data["recommended_ram_gb"] > 0


class TestAddDryRunValidation:
    def test_dry_run_still_rejects_invalid_quant(self, monkeypatch):
        """--dry-run still validates the quant type and exits nonzero for unknown quants."""
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: [])

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "INVALID", "--dry-run"])

        assert result.exit_code != 0
        assert "Unknown quantization" in result.output or "Error" in result.output

    def test_dry_run_still_requires_params_b(self):
        """--dry-run still requires --params-b for non-HF specs."""
        result = runner.invoke(app, ["add", "my-model", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code != 0

    def test_dry_run_still_requires_quant(self):
        """--dry-run still requires --quant for non-HF specs."""
        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--dry-run"])

        assert result.exit_code != 0

    def test_dry_run_with_duplicate_does_not_warn_about_force(self, monkeypatch):
        """--dry-run bypasses the duplicate-model check (it's not writing anyway)."""
        existing = [{
            "id": "my-model", "family": "X", "params_b": 7, "quant": "Q4_K_M",
            "min_vram_gb": 4, "recommended_vram_gb": 5, "recommended_ram_gb": 8, "notes": "",
        }]
        monkeypatch.setattr("llmscan.cli.load_user_catalog", lambda: list(existing))
        monkeypatch.setattr("llmscan.cli.save_user_catalog", lambda e: None)

        result = runner.invoke(app, ["add", "my-model", "--params-b", "7", "--quant", "Q4_K_M", "--dry-run"])

        assert result.exit_code == 0
        assert "--force" not in result.output
