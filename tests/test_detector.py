from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llmscan.detector import (
    GPUInfo,
    MachineProfile,
    _detect_amd_rocm,
    _detect_apple_silicon,
    _detect_intel_gpu,
    _detect_nvidia,
    _detect_ram_gb,
    _detect_windows_gpu,
    _run,
    detect_machine,
)

# ---------------------------------------------------------------------------
# 8.5  MachineProfile.to_dict
# ---------------------------------------------------------------------------


class TestMachineProfileToDict:
    def test_primary_vram_picks_max(self):
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i9",
            ram_gb=32,
            gpus=[
                GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
                GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=12.0, source="nvidia-smi"),
            ],
        )
        d = profile.to_dict()
        assert d["primary_gpu_vram_gb"] == 24.0

    def test_total_vram_sums_all(self):
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i9",
            ram_gb=32,
            gpus=[
                GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
                GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=12.0, source="nvidia-smi"),
            ],
        )
        d = profile.to_dict()
        assert d["total_gpu_vram_gb"] == 36.0

    def test_no_gpus_returns_zero(self):
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=16,
            gpus=[],
        )
        d = profile.to_dict()
        assert d["primary_gpu_vram_gb"] == 0
        assert d["total_gpu_vram_gb"] == 0.0


# ---------------------------------------------------------------------------
# 8.6  _detect_nvidia (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDetectNvidia:
    @patch("llmscan.detector.shutil.which", return_value=None)
    def test_nvidia_smi_not_found(self, mock_which):
        assert _detect_nvidia() == []

    @patch("llmscan.detector._run", return_value="NVIDIA GeForce RTX 4090, 24564")
    @patch("llmscan.detector.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_single_gpu_parsed(self, mock_which, mock_run):
        gpus = _detect_nvidia()
        assert len(gpus) == 1
        assert gpus[0].vendor == "NVIDIA"
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].vram_gb == pytest.approx(24564 / 1024, abs=0.1)

    @patch("llmscan.detector._run", return_value="RTX 3090, 24576\nRTX 3090, 24576")
    @patch("llmscan.detector.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_multi_gpu_parsed(self, mock_which, mock_run):
        gpus = _detect_nvidia()
        assert len(gpus) == 2

    @patch("llmscan.detector._run", return_value="malformed line without comma")
    @patch("llmscan.detector.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_malformed_line_skipped(self, mock_which, mock_run):
        gpus = _detect_nvidia()
        assert len(gpus) == 0

    @patch("llmscan.detector._run", return_value="RTX 4090, N/A")
    @patch("llmscan.detector.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_non_numeric_memory_skipped(self, mock_which, mock_run):
        gpus = _detect_nvidia()
        assert len(gpus) == 0


# ---------------------------------------------------------------------------
# _detect_amd_rocm (mocked)
# ---------------------------------------------------------------------------


class TestDetectAmdRocm:
    @patch("llmscan.detector.shutil.which", return_value=None)
    def test_rocm_smi_not_found(self, mock_which):
        assert _detect_amd_rocm() == []

    @patch("llmscan.detector._run")
    @patch("llmscan.detector.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    def test_single_amd_gpu_parsed(self, mock_which, mock_run):
        csv = "device,Card series,VRAM Total Memory (B)\n0,Radeon RX 7900 XTX,25769803776"
        mock_run.return_value = csv
        gpus = _detect_amd_rocm()
        assert len(gpus) == 1
        assert gpus[0].vendor == "AMD"
        assert gpus[0].name == "Radeon RX 7900 XTX"
        assert gpus[0].vram_gb == pytest.approx(24.0, abs=0.1)
        assert gpus[0].source == "rocm-smi"

    @patch("llmscan.detector._run")
    @patch("llmscan.detector.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    def test_multi_amd_gpu_parsed(self, mock_which, mock_run):
        csv = "device,Card series,VRAM Total Memory (B)\n0,RX 7900 XTX,25769803776\n1,RX 7900 XTX,25769803776"
        mock_run.return_value = csv
        gpus = _detect_amd_rocm()
        assert len(gpus) == 2

    @patch("llmscan.detector._run", return_value=None)
    @patch("llmscan.detector.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    def test_rocm_smi_returns_no_output(self, mock_which, mock_run):
        assert _detect_amd_rocm() == []

    @patch("llmscan.detector._run")
    @patch("llmscan.detector.shutil.which", return_value="/opt/rocm/bin/rocm-smi")
    def test_malformed_csv_line_skipped(self, mock_which, mock_run):
        csv = "device,Card series,VRAM Total Memory (B)\nbadline"
        mock_run.return_value = csv
        gpus = _detect_amd_rocm()
        assert len(gpus) == 0


# ---------------------------------------------------------------------------
# _detect_intel_gpu (mocked)
# ---------------------------------------------------------------------------


class TestDetectIntelGpu:
    @patch("llmscan.detector.shutil.which", return_value=None)
    def test_no_tools_available(self, mock_which):
        assert _detect_intel_gpu() == []

    @patch("llmscan.detector._run")
    @patch("llmscan.detector.shutil.which")
    def test_xpu_smi_parses_gpu(self, mock_which, mock_run):
        def which_side(cmd):
            return "/usr/bin/xpu-smi" if cmd == "xpu-smi" else None

        mock_which.side_effect = which_side
        csv = "DeviceName,Memory Physical Size (MiB)\nIntel Arc A770,16384"
        mock_run.return_value = csv
        gpus = _detect_intel_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "Intel"
        assert gpus[0].name == "Intel Arc A770"
        assert gpus[0].vram_gb == pytest.approx(16.0, abs=0.1)
        assert gpus[0].source == "xpu-smi"

    @patch("llmscan.detector._run")
    @patch("llmscan.detector.shutil.which")
    def test_clinfo_fallback(self, mock_which, mock_run):
        def which_side(cmd):
            if cmd == "xpu-smi":
                return None
            if cmd == "clinfo":
                return "/usr/bin/clinfo"
            return None

        mock_which.side_effect = which_side
        clinfo_out = (
            "  CL_DEVICE_NAME                          Intel(R) Arc(TM) A770\n"
            "  CL_DEVICE_GLOBAL_MEM_SIZE                17179869184\n"
        )
        mock_run.return_value = clinfo_out
        gpus = _detect_intel_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "Intel"
        assert gpus[0].vram_gb == pytest.approx(16.0, abs=0.1)
        assert gpus[0].source == "clinfo"

    @patch("llmscan.detector._run", return_value=None)
    @patch("llmscan.detector.shutil.which")
    def test_xpu_smi_no_output_falls_through(self, mock_which, mock_run):
        def which_side(cmd):
            return "/usr/bin/xpu-smi" if cmd == "xpu-smi" else None

        mock_which.side_effect = which_side
        gpus = _detect_intel_gpu()
        assert gpus == []


# ---------------------------------------------------------------------------
# 8.7  _detect_apple_silicon (mocked)
# ---------------------------------------------------------------------------


class TestDetectAppleSilicon:
    @patch("llmscan.detector.platform.system", return_value="Linux")
    def test_non_darwin_returns_empty(self, mock_sys):
        gpus, unified = _detect_apple_silicon()
        assert gpus == []
        assert unified is None

    @patch("llmscan.detector.platform.machine", return_value="arm64")
    @patch("llmscan.detector.platform.system", return_value="Darwin")
    @patch("llmscan.detector._run")
    def test_apple_m_series_chip(self, mock_run, mock_sys, mock_mach):
        def run_side_effect(cmd):
            if "brand_string" in cmd:
                return "Apple M2 Max"
            if "hw.memsize" in cmd:
                return str(32 * 1024**3)  # 32 GB
            return None

        mock_run.side_effect = run_side_effect

        gpus, unified = _detect_apple_silicon()
        assert len(gpus) == 1
        assert gpus[0].vendor == "Apple"
        assert gpus[0].vram_gb == pytest.approx(32 * 0.65, abs=0.1)
        assert unified == 32.0

    @patch("llmscan.detector.platform.processor", return_value="arm")
    @patch("llmscan.detector.platform.machine", return_value="arm64")
    @patch("llmscan.detector.platform.system", return_value="Darwin")
    @patch("llmscan.detector._run")
    def test_non_apple_arm64_still_detected(self, mock_run, mock_sys, mock_mach, mock_proc):
        def run_side_effect(cmd):
            if "brand_string" in cmd:
                return None
            if "hw.memsize" in cmd:
                return str(16 * 1024**3)
            return None

        mock_run.side_effect = run_side_effect

        # platform.processor() returns "arm", no "Apple" in it,
        # but platform.machine() == "arm64" should still trigger detection.
        gpus, unified = _detect_apple_silicon()
        assert len(gpus) == 1
        assert unified == 16.0


# ---------------------------------------------------------------------------
# 8.8  _detect_windows_gpu (mocked)
# ---------------------------------------------------------------------------


class TestDetectWindowsGpu:
    @patch("llmscan.detector.platform.system", return_value="Linux")
    def test_non_windows_returns_empty(self, mock_sys):
        assert _detect_windows_gpu() == []

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run")
    def test_powershell_path_preferred(self, mock_run, mock_sys):
        """PowerShell output is tried first; wmic is not called."""
        mock_run.return_value = "Name=NVIDIA GeForce RTX 3070\nAdapterRAM=8589934592\n"
        gpus = _detect_windows_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "NVIDIA"
        assert gpus[0].source == "powershell"
        assert gpus[0].vram_gb == pytest.approx(8.0, abs=0.1)

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run")
    def test_wmic_fallback_when_powershell_fails(self, mock_run, mock_sys):
        """When PowerShell returns None, falls back to wmic."""
        wmic_out = "AdapterRAM=8589934592\nName=NVIDIA GeForce RTX 3070"
        # First call (powershell) → None, second call (wmic) → output
        mock_run.side_effect = [None, wmic_out]
        gpus = _detect_windows_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "NVIDIA"
        assert gpus[0].source == "wmic"

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run")
    def test_amd_vendor_detection(self, mock_run, mock_sys):
        mock_run.return_value = "AdapterRAM=8589934592\nName=AMD Radeon RX 7900\n"
        gpus = _detect_windows_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "AMD"

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run")
    def test_unknown_vendor(self, mock_run, mock_sys):
        mock_run.return_value = "AdapterRAM=4294967296\nName=Matrox G200\n"
        gpus = _detect_windows_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "Unknown"

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run")
    def test_missing_adapter_ram(self, mock_run, mock_sys):
        mock_run.return_value = "Name=Some GPU\n"
        gpus = _detect_windows_gpu()
        assert len(gpus) == 1
        assert gpus[0].vram_gb == 0.0

    @patch("llmscan.detector.platform.system", return_value="Windows")
    @patch("llmscan.detector._run", return_value=None)
    def test_both_paths_fail_returns_empty(self, mock_run, mock_sys):
        gpus = _detect_windows_gpu()
        assert gpus == []


# ---------------------------------------------------------------------------
# 8.9  _detect_ram_gb (mocked)
# ---------------------------------------------------------------------------


class TestDetectRamGb:
    def test_linux_sysconf_path(self):
        """On this Linux box, sysconf should work and return > 0."""
        result = _detect_ram_gb()
        assert isinstance(result, float)
        assert result > 0

    @patch("llmscan.detector.platform.system", return_value="Linux")
    def test_linux_proc_meminfo_fallback(self, mock_sys):
        meminfo = "MemTotal:       32878580 kB\nMemFree:        1234567 kB\n"

        import os as _os

        # Force the sysconf branch to be attempted, then fail into /proc/meminfo.
        with (
            patch.object(_os, "sysconf_names", {"SC_PAGE_SIZE": 1, "SC_PHYS_PAGES": 1}, create=True),
            patch.object(_os, "sysconf", MagicMock(side_effect=OSError("forced failure")), create=True),
            patch("pathlib.Path.read_text", return_value=meminfo),
        ):
            result = _detect_ram_gb()
            assert result == pytest.approx(32878580 / (1024**2), abs=0.1)

    @patch("llmscan.detector.platform.system", return_value="Windows")
    def test_windows_powershell_ram_path(self, mock_sys):
        import os as _os

        # PowerShell returns raw bytes as a string
        with (
            patch.object(_os, "sysconf_names", {"SC_PAGE_SIZE": 1, "SC_PHYS_PAGES": 1}, create=True),
            patch.object(_os, "sysconf", MagicMock(side_effect=OSError("forced failure")), create=True),
            patch("llmscan.detector._run", return_value="17179869184"),
        ):
            result = _detect_ram_gb()
            assert result == pytest.approx(16.0, abs=0.1)

    @patch("llmscan.detector.platform.system", return_value="Windows")
    def test_windows_wmic_fallback_path(self, mock_sys):
        import os as _os

        # PowerShell fails, wmic succeeds
        wmic_out = "TotalPhysicalMemory=17179869184"
        with (
            patch.object(_os, "sysconf_names", {"SC_PAGE_SIZE": 1, "SC_PHYS_PAGES": 1}, create=True),
            patch.object(_os, "sysconf", MagicMock(side_effect=OSError("forced failure")), create=True),
            patch("llmscan.detector._run", side_effect=[None, wmic_out]),
        ):
            result = _detect_ram_gb()
            assert result == pytest.approx(16.0, abs=0.1)

    @patch("llmscan.detector.platform.system", return_value="FreeBSD")
    def test_all_paths_fail_returns_zero(self, mock_sys):
        import os as _os

        with (
            patch.object(_os, "sysconf_names", {"SC_PAGE_SIZE": 1, "SC_PHYS_PAGES": 1}, create=True),
            patch.object(_os, "sysconf", MagicMock(side_effect=OSError("forced failure")), create=True),
        ):
            result = _detect_ram_gb()
            assert result == 0.0


# ---------------------------------------------------------------------------
# 8.10  detect_machine GPU collapsing
# ---------------------------------------------------------------------------


class TestGpuCollapsing:
    @patch("llmscan.detector._detect_windows_gpu", return_value=[])
    @patch("llmscan.detector._detect_apple_silicon", return_value=([], None))
    @patch("llmscan.detector._detect_intel_gpu", return_value=[])
    @patch("llmscan.detector._detect_amd_rocm", return_value=[])
    @patch("llmscan.detector._detect_nvidia")
    @patch("llmscan.detector._detect_ram_gb", return_value=32.0)
    def test_identical_gpus_collapsed(self, mock_ram, mock_nv, mock_amd, mock_intel, mock_apple, mock_win):
        mock_nv.return_value = [
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
        ]
        profile = detect_machine()
        assert len(profile.gpus) == 1
        assert profile.gpus[0].count == 4

    @patch("llmscan.detector._detect_windows_gpu", return_value=[])
    @patch("llmscan.detector._detect_apple_silicon", return_value=([], None))
    @patch("llmscan.detector._detect_intel_gpu", return_value=[])
    @patch("llmscan.detector._detect_amd_rocm", return_value=[])
    @patch("llmscan.detector._detect_nvidia")
    @patch("llmscan.detector._detect_ram_gb", return_value=32.0)
    def test_different_gpus_kept_separate(self, mock_ram, mock_nv, mock_amd, mock_intel, mock_apple, mock_win):
        mock_nv.return_value = [
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, source="nvidia-smi"),
            GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi"),
        ]
        profile = detect_machine()
        assert len(profile.gpus) == 2
        assert all(g.count == 1 for g in profile.gpus)


# ---------------------------------------------------------------------------
# 8.11  _run helper
# ---------------------------------------------------------------------------


class TestRunHelper:
    @patch("llmscan.detector.subprocess.run")
    def test_successful_command(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0, stdout="  hello world  ")
        result = _run(["echo", "hello"])
        assert result == "hello world"

    @patch("llmscan.detector.subprocess.run")
    def test_failed_command_returns_none(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=1, stdout="error")
        result = _run(["false"])
        assert result is None

    @patch("llmscan.detector.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_binary_returns_none(self, mock_subproc):
        result = _run(["nonexistent_binary"])
        assert result is None

    @patch("llmscan.detector.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["hang"], timeout=10))
    def test_timeout_returns_none(self, mock_subproc):
        result = _run(["hang"])
        assert result is None

    @patch("llmscan.detector.subprocess.run")
    def test_timeout_kwarg_is_passed(self, mock_subproc):
        mock_subproc.return_value = MagicMock(returncode=0, stdout="ok")
        _run(["echo"])
        _, kwargs = mock_subproc.call_args
        assert "timeout" in kwargs
        assert kwargs["timeout"] > 0

    @patch("llmscan.detector.subprocess.run", side_effect=PermissionError("denied"))
    def test_permission_error_logged_and_returns_none(self, mock_subproc, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="llmscan.detector"):
            result = _run(["nvidia-smi"])
        assert result is None
        assert "denied" in caplog.text
