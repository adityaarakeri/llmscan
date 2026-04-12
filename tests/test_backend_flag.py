from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.estimator import evaluate_models

runner = CliRunner()

# Machine with exactly 6.0 GB GPU — at the rec_vram threshold for our test model
EXACT_FIT = MachineProfile(
    os="Linux", arch="x86_64", cpu="i9", ram_gb=12.0,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=6.0, source="nvidia-smi")],
)

_CATALOG = [
    {
        "id": "test-6b",
        "family": "Test",
        "params_b": 6,
        "quant": "Q4_K_M",
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 6.0,
        "recommended_ram_gb": 9.0,
        "notes": "",
    },
]


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


def _invoke(args, catalog=_CATALOG):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(catalog, f)
        name = f.name
    try:
        with patch.object(cli_module, "_get_profile", return_value=EXACT_FIT):
            result = runner.invoke(app, args + ["--catalog", name])
    finally:
        os.unlink(name)
    return result


class TestBackendFlagAccepted:
    def test_llama_cpp_backend_accepted(self):
        """--backend llama-cpp is accepted and exits 0."""
        result = _invoke(["list", "--backend", "llama-cpp", "--json"])
        assert result.exit_code == 0

    def test_ollama_backend_accepted(self):
        """--backend ollama is accepted and exits 0."""
        result = _invoke(["list", "--backend", "ollama", "--json"])
        assert result.exit_code == 0

    def test_mlx_backend_accepted(self):
        """--backend mlx is accepted and exits 0."""
        result = _invoke(["list", "--backend", "mlx", "--json"])
        assert result.exit_code == 0

    def test_invalid_backend_returns_nonzero_exit(self):
        """--backend with an unsupported value exits non-zero."""
        result = _invoke(["list", "--backend", "vllm", "--json"])
        assert result.exit_code != 0

    def test_invalid_backend_error_message_mentions_valid_options(self):
        """Error for invalid --backend names the valid backends."""
        result = _invoke(["list", "--backend", "vllm"])
        output = result.output.lower()
        assert "ollama" in output or "llama-cpp" in output or "mlx" in output


class TestBackendScoringDifference:
    def test_llama_cpp_scores_great_at_exact_rec_vram(self):
        """With llama-cpp, a GPU at exact rec_vram scores 'great'."""
        result = _invoke(["list", "--backend", "llama-cpp", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        model = next(m for m in data["models"] if m["id"] == "test-6b")
        assert model["rating"] == "great"

    def test_ollama_downgrades_from_great_due_to_overhead(self):
        """Ollama's VRAM overhead means a model at exact rec_vram no longer scores 'great'."""
        result = _invoke(["list", "--backend", "ollama", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        model = next(m for m in data["models"] if m["id"] == "test-6b")
        assert model["rating"] != "great"

    def test_mlx_downgrades_from_great_due_to_overhead(self):
        """mlx backend overhead means a model at exact rec_vram no longer scores 'great'."""
        result = _invoke(["list", "--backend", "mlx", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        model = next(m for m in data["models"] if m["id"] == "test-6b")
        assert model["rating"] != "great"

    def test_ollama_rating_is_lower_than_llama_cpp_for_exact_fit(self):
        """Ollama scores worse than llama-cpp when hardware exactly meets rec_vram."""
        r_llama = _invoke(["list", "--backend", "llama-cpp", "--min-rating", "no", "--json"])
        r_ollama = _invoke(["list", "--backend", "ollama", "--min-rating", "no", "--json"])
        data_llama = json.loads(r_llama.output)
        data_ollama = json.loads(r_ollama.output)
        from llmscan.estimator import RATING_ORDER
        m_llama = next(m for m in data_llama["models"] if m["id"] == "test-6b")
        m_ollama = next(m for m in data_ollama["models"] if m["id"] == "test-6b")
        assert RATING_ORDER[m_ollama["rating"]] < RATING_ORDER[m_llama["rating"]]


class TestBackendJsonOutput:
    def test_json_includes_backend_field(self):
        """JSON output includes the backend used for scoring."""
        result = _invoke(["list", "--backend", "ollama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "backend" in data
        assert data["backend"] == "ollama"

    def test_default_backend_in_json_is_llama_cpp(self):
        """Without --backend, JSON output reports llama-cpp as the backend."""
        result = _invoke(["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data.get("backend") == "llama-cpp"

    def test_mlx_backend_reflected_in_json_output(self):
        """--backend mlx is reported correctly in JSON output."""
        result = _invoke(["list", "--backend", "mlx", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["backend"] == "mlx"


class TestBackendFitNotesAndReasonCode:
    def test_backend_note_in_fit_notes_for_ollama(self):
        """Ollama backend appends a context note to fit_notes for scored models."""
        result = _invoke(["list", "--backend", "ollama", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        model = next(m for m in data["models"] if m["id"] == "test-6b")
        assert "ollama" in model["fit_notes"].lower()

    def test_no_backend_note_for_llama_cpp_default(self):
        """Default llama-cpp backend does not add extra overhead notes to fit_notes."""
        result = _invoke(["list", "--backend", "llama-cpp", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        model = next(m for m in data["models"] if m["id"] == "test-6b")
        assert "overhead" not in model["fit_notes"].lower()


class TestEvaluateModelsBackendParam:
    def test_evaluate_models_accepts_backend_param(self):
        """evaluate_models() accepts a backend keyword argument."""
        rows = evaluate_models(EXACT_FIT, _CATALOG, backend="ollama")
        assert len(rows) == 1

    def test_evaluate_models_llama_cpp_scores_great(self):
        """evaluate_models with llama-cpp backend scores 'great' at exact rec_vram."""
        rows = evaluate_models(EXACT_FIT, _CATALOG, backend="llama-cpp")
        assert rows[0]["rating"] == "great"

    def test_evaluate_models_ollama_scores_lower_than_llama_cpp(self):
        """evaluate_models with ollama backend scores lower than llama-cpp for exact-fit model."""
        from llmscan.estimator import RATING_ORDER
        rows_llama = evaluate_models(EXACT_FIT, _CATALOG, backend="llama-cpp")
        rows_ollama = evaluate_models(EXACT_FIT, _CATALOG, backend="ollama")
        assert RATING_ORDER[rows_ollama[0]["rating"]] < RATING_ORDER[rows_llama[0]["rating"]]

    def test_evaluate_models_default_backend_is_llama_cpp(self):
        """evaluate_models() without backend kwarg behaves identically to llama-cpp."""
        rows_default = evaluate_models(EXACT_FIT, _CATALOG)
        rows_llama = evaluate_models(EXACT_FIT, _CATALOG, backend="llama-cpp")
        assert rows_default[0]["rating"] == rows_llama[0]["rating"]
