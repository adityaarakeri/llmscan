from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.estimator import _score_model, evaluate_models

runner = CliRunner()

# ---------------------------------------------------------------------------
# Profiles that trigger each scoring branch
# ---------------------------------------------------------------------------

_MODEL = {
    "id": "test-8b",
    "family": "Test",
    "params_b": 8,
    "quant": "Q4_K_M",
    "min_vram_gb": 5.5,
    "recommended_vram_gb": 8.0,
    "recommended_ram_gb": 16.0,
    "notes": "",
}

GREAT_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i9",
    ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)
OK_SINGLE_GPU_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i7",
    ram_gb=14,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=6.0, source="nvidia-smi")],
)
OK_MULTI_GPU_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Xeon",
    ram_gb=32,
    gpus=[
        GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=6.0, source="nvidia-smi"),
        GPUInfo(vendor="NVIDIA", name="RTX 3060 Ti", vram_gb=6.0, source="nvidia-smi"),
    ],
)
OK_CPU_ONLY_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i9",
    ram_gb=64,
    gpus=[],
)
TIGHT_PARTIAL_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i7",
    ram_gb=16,
    gpus=[GPUInfo(vendor="NVIDIA", name="GTX 1650", vram_gb=4.0, source="nvidia-smi")],
)
TIGHT_MULTI_GPU_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Xeon",
    ram_gb=16,
    gpus=[
        GPUInfo(vendor="NVIDIA", name="GTX 1060", vram_gb=3.0, source="nvidia-smi"),
        GPUInfo(vendor="NVIDIA", name="GTX 1060", vram_gb=3.0, source="nvidia-smi"),
    ],
)
NO_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Celeron",
    ram_gb=4,
    gpus=[GPUInfo(vendor="NVIDIA", name="GT 730", vram_gb=1.0, source="nvidia-smi")],
)


# ---------------------------------------------------------------------------
# _score_model returns a 3-tuple: (rating, reason_code, fit_notes)
# ---------------------------------------------------------------------------


class TestScoreModelReturnsReasonCode:
    def test_returns_three_values(self):
        """_score_model must return a 3-tuple: (rating, reason_code, fit_notes)."""
        result = _score_model(GREAT_PROFILE, _MODEL)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"

    def test_great_reason_code_is_empty_string(self):
        """great rating has an empty reason code — the label speaks for itself."""
        _, reason_code, _ = _score_model(GREAT_PROFILE, _MODEL)
        assert reason_code == ""

    def test_ok_single_gpu_reason_code_contains_vram_deficit(self):
        """ok via single GPU below recommended shows vram deficit percentage."""
        _, reason_code, _ = _score_model(OK_SINGLE_GPU_PROFILE, _MODEL)
        # GPU is 6 GB, recommended is 8 GB → 25% below
        assert "vram" in reason_code.lower()
        assert "%" in reason_code

    def test_ok_single_gpu_deficit_percentage_is_correct(self):
        """VRAM deficit percentage matches (rec - gpu) / rec * 100, rounded."""
        # GPU=6, rec=8 → (8-6)/8*100 = 25%
        _, reason_code, _ = _score_model(OK_SINGLE_GPU_PROFILE, _MODEL)
        assert "25" in reason_code

    def test_ok_multi_gpu_reason_code_is_multi_gpu(self):
        """ok via tensor parallelism has reason_code 'multi-gpu'."""
        _, reason_code, _ = _score_model(OK_MULTI_GPU_PROFILE, _MODEL)
        assert "multi-gpu" in reason_code.lower() or "multi gpu" in reason_code.lower()

    def test_ok_cpu_only_reason_code_is_cpu_only(self):
        """ok via CPU inference has reason_code 'cpu-only'."""
        _, reason_code, _ = _score_model(OK_CPU_ONLY_PROFILE, _MODEL)
        assert "cpu" in reason_code.lower()

    def test_tight_partial_offload_reason_code(self):
        """tight via partial offload has reason_code mentioning offload."""
        _, reason_code, _ = _score_model(TIGHT_PARTIAL_PROFILE, _MODEL)
        assert "offload" in reason_code.lower() or "partial" in reason_code.lower()

    def test_tight_multi_gpu_reason_code(self):
        """tight via multi-GPU minimum has reason_code mentioning multi-gpu."""
        _, reason_code, _ = _score_model(TIGHT_MULTI_GPU_PROFILE, _MODEL)
        assert "multi" in reason_code.lower()

    def test_no_reason_code_is_empty_string(self):
        """no rating has an empty reason code."""
        _, reason_code, _ = _score_model(NO_PROFILE, _MODEL)
        assert reason_code == ""

    def test_fit_notes_still_returned_as_third_element(self):
        """fit_notes (the long explanation) is still the third return value."""
        _, _, fit_notes = _score_model(GREAT_PROFILE, _MODEL)
        assert "GPU VRAM meets recommended target" in fit_notes


# ---------------------------------------------------------------------------
# evaluate_models includes reason_code in each row dict
# ---------------------------------------------------------------------------


class TestEvaluateModelsReasonCode:
    def test_rows_include_reason_code_key(self):
        rows = evaluate_models(GREAT_PROFILE, [_MODEL])
        assert "reason_code" in rows[0]

    def test_reason_code_is_string(self):
        for profile in [GREAT_PROFILE, OK_SINGLE_GPU_PROFILE, NO_PROFILE]:
            rows = evaluate_models(profile, [_MODEL])
            assert isinstance(rows[0]["reason_code"], str)

    def test_reason_code_matches_score_model(self):
        """evaluate_models passes reason_code through from _score_model unchanged."""
        _, expected_code, _ = _score_model(OK_CPU_ONLY_PROFILE, _MODEL)
        rows = evaluate_models(OK_CPU_ONLY_PROFILE, [_MODEL])
        assert rows[0]["reason_code"] == expected_code


# ---------------------------------------------------------------------------
# list command renders reason codes in the Fit column
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestListRendersReasonCode:
    def test_list_shows_cpu_only_reason_in_output(self):
        """list output includes 'cpu' when the machine is CPU-only."""
        with patch.object(cli_module, "_get_profile", return_value=OK_CPU_ONLY_PROFILE):
            result = runner.invoke(app, ["list", "--min-rating", "no"])
        assert result.exit_code == 0
        assert "cpu" in result.output.lower()

    def test_list_shows_multi_gpu_reason_in_output(self):
        """list output includes 'multi' when multi-GPU scoring fires."""
        with patch.object(cli_module, "_get_profile", return_value=OK_MULTI_GPU_PROFILE):
            result = runner.invoke(app, ["list", "--min-rating", "no"])
        assert result.exit_code == 0
        assert "multi" in result.output.lower()

    def test_list_json_includes_reason_code_field(self):
        """list --json output has reason_code in each model row."""
        with patch.object(cli_module, "_get_profile", return_value=GREAT_PROFILE):
            result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for model in data["models"]:
            assert "reason_code" in model, f"Model {model.get('id')} missing reason_code"

    def test_list_json_reason_code_is_string(self):
        """reason_code in list --json is a string for all rated models."""
        with patch.object(cli_module, "_get_profile", return_value=GREAT_PROFILE):
            result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for model in data["models"]:
            assert isinstance(model["reason_code"], str)
