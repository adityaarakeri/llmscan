from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

DUAL_GPU = MachineProfile(
    os="Linux", arch="x86_64", cpu="Xeon", ram_gb=128,
    gpus=[
        GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, count=2, source="nvidia-smi"),
    ],
)

MIXED_GPU = MachineProfile(
    os="Linux", arch="x86_64", cpu="Xeon", ram_gb=128,
    gpus=[
        GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, count=1, source="nvidia-smi"),
        GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=12.0, count=1, source="nvidia-smi"),
    ],
)

SINGLE_GPU = MachineProfile(
    os="Linux", arch="x86_64", cpu="i9", ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestScanMultiGpuDisplay:
    def test_scan_shows_total_vram_for_multi_gpu(self):
        """scan output shows total VRAM when multiple GPU units are present."""
        with patch.object(cli_module, "_get_profile", return_value=DUAL_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "48" in result.output  # 2 x 24 GB = 48 GB total

    def test_scan_shows_total_vram_for_mixed_gpu(self):
        """scan shows total VRAM for mixed multi-GPU setups."""
        with patch.object(cli_module, "_get_profile", return_value=MIXED_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "36" in result.output  # 24 + 12 = 36 GB total

    def test_scan_no_total_vram_for_single_gpu(self):
        """scan does not show a 'total' VRAM line for a single GPU."""
        with patch.object(cli_module, "_get_profile", return_value=SINGLE_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "total" not in result.output.lower()

    def test_scan_shows_per_gpu_count(self):
        """scan shows the x2 count for identical collapsed GPUs."""
        with patch.object(cli_module, "_get_profile", return_value=DUAL_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "x2" in result.output or "×2" in result.output or "2x" in result.output or "2 ×" in result.output

    def test_scan_shows_individual_vram_per_gpu(self):
        """scan shows per-card VRAM for each GPU in a mixed setup."""
        with patch.object(cli_module, "_get_profile", return_value=MIXED_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "24" in result.output
        assert "12" in result.output


class TestScanJsonMultiGpu:
    def test_scan_json_includes_gpu_count(self):
        """scan --json includes count field for each GPU entry."""
        with patch.object(cli_module, "_get_profile", return_value=DUAL_GPU):
            result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        gpu = data["gpus"][0]
        assert "count" in gpu
        assert gpu["count"] == 2

    def test_scan_json_includes_total_vram(self):
        """scan --json includes total_gpu_vram_gb for multi-GPU."""
        with patch.object(cli_module, "_get_profile", return_value=DUAL_GPU):
            result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_gpu_vram_gb" in data
        assert data["total_gpu_vram_gb"] == pytest.approx(48.0)

    def test_scan_json_includes_primary_vram(self):
        """scan --json includes primary_gpu_vram_gb (max single card)."""
        with patch.object(cli_module, "_get_profile", return_value=MIXED_GPU):
            result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["primary_gpu_vram_gb"] == pytest.approx(24.0)

    def test_scan_json_mixed_gpus_all_listed(self):
        """scan --json lists all distinct GPU entries for mixed setups."""
        with patch.object(cli_module, "_get_profile", return_value=MIXED_GPU):
            result = runner.invoke(app, ["scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["gpus"]) == 2
        vrams = {g["vram_gb"] for g in data["gpus"]}
        assert 24.0 in vrams
        assert 12.0 in vrams


class TestScanSummaryPanelMultiGpu:
    def test_summary_panel_shows_total_vram_badge_for_multi_gpu(self):
        """The hardware summary panel shows a TOTAL VRAM badge for multi-GPU."""
        with patch.object(cli_module, "_get_profile", return_value=DUAL_GPU):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        assert "TOTAL" in result.output.upper() or "48" in result.output
