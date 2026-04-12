from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

FAKE_PROFILE_NO_GPU = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Test CPU",
    ram_gb=32,
    gpus=[],
)

FAKE_PROFILE_GPU_NO_VRAM = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Test CPU",
    ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3080", vram_gb=0.0, source="nvidia-smi")],
)

FAKE_PROFILE_HEALTHY = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Test CPU",
    ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)


@pytest.fixture(autouse=True)
def _reset_cache():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestDoctorCommand:
    def test_doctor_exits_successfully(self):
        """llmscan doctor should exit with code 0."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_HEALTHY):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_shows_all_six_tools(self):
        """Doctor reports availability of all six hardware detection tools."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_NO_GPU):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        for tool in ("nvidia-smi", "rocm-smi", "xpu-smi", "sysctl", "wmic", "clinfo"):
            assert tool in result.output

    def test_doctor_marks_available_tool_as_found(self):
        """When nvidia-smi is on PATH, doctor marks it as found."""

        def which_side(cmd):
            return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None

        with (
            patch("shutil.which", side_effect=which_side),
            patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_NO_GPU),
        ):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "nvidia-smi" in result.output
        # Should show found/available indicator
        assert "found" in result.output.lower() or "✓" in result.output or "ok" in result.output.lower()

    def test_doctor_marks_missing_tool_as_not_found(self):
        """When no tools are on PATH, doctor marks them all as missing."""
        with (
            patch("shutil.which", return_value=None),
            patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_NO_GPU),
        ):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Should show missing/not-found indicator
        assert "missing" in result.output.lower() or "not found" in result.output.lower() or "✗" in result.output

    def test_doctor_flags_gpu_with_zero_vram_as_anomaly(self):
        """Doctor flags a detected GPU with 0 VRAM as an anomaly."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_GPU_NO_VRAM):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Should mention VRAM issue or anomaly
        assert "0" in result.output
        assert (
            "anomaly" in result.output.lower() or "warning" in result.output.lower() or "vram" in result.output.lower()
        )

    def test_doctor_no_anomaly_for_healthy_gpu(self):
        """Doctor does not report a VRAM anomaly when GPU has nonzero VRAM."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_HEALTHY):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "anomaly" not in result.output.lower()

    def test_doctor_reports_no_gpu_when_none_detected(self):
        """Doctor tells the user no GPU was detected when gpus list is empty."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_NO_GPU):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        out = result.output.lower()
        assert "no gpu" in out or "no dedicated gpu" in out or "cpu-only" in out

    def test_doctor_json_output_contains_tools_key(self):
        """Doctor --json output has a 'tools' key listing each tool's availability."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_HEALTHY):
            result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "tools" in data
        assert isinstance(data["tools"], dict)

    def test_doctor_json_output_contains_anomalies_key(self):
        """Doctor --json output has an 'anomalies' key listing issues found."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_GPU_NO_VRAM):
            result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    def test_doctor_json_output_zero_vram_anomaly_is_reported(self):
        """Doctor --json anomalies list includes an entry for GPU with 0 VRAM."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_GPU_NO_VRAM):
            result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["anomalies"]) >= 1
        combined = " ".join(data["anomalies"]).lower()
        assert "vram" in combined or "0" in combined

    def test_doctor_json_tool_availability_is_boolean(self):
        """Each tool in --json tools dict maps to a boolean available field."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_HEALTHY):
            result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for tool_name, tool_info in data["tools"].items():
            assert "available" in tool_info, f"Tool {tool_name!r} missing 'available' key"
            assert isinstance(tool_info["available"], bool)

    def test_doctor_json_no_anomalies_when_gpu_healthy(self):
        """Doctor --json anomalies list is empty when all detections look correct."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE_HEALTHY):
            result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["anomalies"] == []
