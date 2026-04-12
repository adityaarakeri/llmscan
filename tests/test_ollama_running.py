from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

STRONG = MachineProfile(
    os="Linux", arch="x86_64", cpu="i9", ram_gb=128,
    gpus=[GPUInfo(vendor="NVIDIA", name="H100", vram_gb=80.0, source="nvidia-smi")],
)

_CATALOG = [
    {"id": "llama-3.1-8b-instruct", "family": "Llama", "params_b": 8, "quant": "Q4_K_M",
     "min_vram_gb": 5.0, "recommended_vram_gb": 6.0, "recommended_ram_gb": 10.0, "notes": ""},
    {"id": "mistral-7b", "family": "Mistral", "params_b": 7, "quant": "Q4_K_M",
     "min_vram_gb": 4.5, "recommended_vram_gb": 5.5, "recommended_ram_gb": 9.0, "notes": ""},
]

# Ollama /api/tags response — llama is running, mistral is not
_OLLAMA_TAGS = {
    "models": [
        {"name": "llama3.1:8b", "model": "llama3.1:8b"},
        {"name": "llama3.1:8b-instruct-q4_K_M", "model": "llama3.1:8b-instruct-q4_K_M"},
    ]
}


def _mock_ollama_resp(data=_OLLAMA_TAGS, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def _invoke(args, catalog=_CATALOG):
    import json as _j, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _j.dump(catalog, f)
        name = f.name
    try:
        with patch.object(cli_module, "_get_profile", return_value=STRONG):
            result = runner.invoke(app, args + ["--catalog", name])
    finally:
        os.unlink(name)
    return result


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestRunningFlagAccepted:
    def test_running_flag_accepted_without_error_when_ollama_up(self):
        """--running flag is accepted; exits 0 when Ollama responds."""
        with patch("httpx.get", return_value=_mock_ollama_resp()):
            result = _invoke(["list", "--running"])
        assert result.exit_code == 0

    def test_running_flag_accepted_when_ollama_down(self):
        """--running exits 0 and warns when Ollama is not reachable."""
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = _invoke(["list", "--running"])
        assert result.exit_code == 0

    def test_running_flag_shows_warning_when_ollama_down(self):
        """When Ollama is not reachable, output warns the user."""
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = _invoke(["list", "--running"])
        assert "ollama" in result.output.lower()


class TestRunningColumnDisplay:
    def test_running_column_appears_in_table(self):
        """--running adds 'running' field to JSON output."""
        with patch("httpx.get", return_value=_mock_ollama_resp()):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # At least one model should have the running key
        assert any("running" in m for m in data["models"])

    def test_running_model_marked_in_output(self):
        """A model available in Ollama has running=True in JSON output."""
        ollama_data = {"models": [{"name": "llama-3.1-8b-instruct:latest"}]}
        with patch("httpx.get", return_value=_mock_ollama_resp(ollama_data)):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        llama = next((m for m in data["models"] if m["id"] == "llama-3.1-8b-instruct"), None)
        assert llama is not None
        assert llama["running"] is True

    def test_json_running_field_present_when_flag_used(self):
        """With --running --json, each model row includes a 'running' boolean field."""
        with patch("httpx.get", return_value=_mock_ollama_resp()):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for model in data["models"]:
            assert "running" in model, f"Model {model.get('id')} missing 'running' field"
            assert isinstance(model["running"], bool)

    def test_json_running_false_without_flag(self):
        """Without --running, JSON output does not include 'running' field."""
        result = _invoke(["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for model in data["models"]:
            assert "running" not in model


class TestOllamaNameMatching:
    def test_exact_id_match_detected_as_running(self):
        """A catalog model whose ID appears in an Ollama model name is marked running."""
        ollama_data = {"models": [{"name": "llama-3.1-8b-instruct:latest"}]}
        with patch("httpx.get", return_value=_mock_ollama_resp(ollama_data)):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        llama = next(m for m in data["models"] if m["id"] == "llama-3.1-8b-instruct")
        assert llama["running"] is True

    def test_unrelated_model_not_marked_running(self):
        """A catalog model with no Ollama match is marked running=False."""
        ollama_data = {"models": [{"name": "llama-3.1-8b-instruct:latest"}]}
        with patch("httpx.get", return_value=_mock_ollama_resp(ollama_data)):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        mistral = next(m for m in data["models"] if m["id"] == "mistral-7b")
        assert mistral["running"] is False

    def test_no_ollama_models_all_false(self):
        """When Ollama returns an empty model list, all catalog models are running=False."""
        ollama_data = {"models": []}
        with patch("httpx.get", return_value=_mock_ollama_resp(ollama_data)):
            result = _invoke(["list", "--running", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(not m["running"] for m in data["models"])
