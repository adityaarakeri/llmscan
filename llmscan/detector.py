from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 10  # seconds


@dataclass
class GPUInfo:
    vendor: str
    name: str
    vram_gb: float
    count: int = 1
    source: str = "unknown"


@dataclass
class MachineProfile:
    os: str
    arch: str
    cpu: str
    ram_gb: float
    gpus: list[GPUInfo]
    unified_memory_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primary_gpu_vram_gb"] = max((g.vram_gb for g in self.gpus), default=0)
        data["total_gpu_vram_gb"] = round(sum(g.vram_gb for g in self.gpus), 1)
        return data


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        log.debug("Command %s exited with code %d", cmd, result.returncode)
    except subprocess.TimeoutExpired:
        log.debug("Command %s timed out after %ds", cmd, _SUBPROCESS_TIMEOUT)
    except Exception as exc:
        log.debug("Command %s failed: %s", cmd, exc)
    return None


def _detect_ram_gb() -> float:
    try:
        import os

        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return round((page_size * pages) / (1024**3), 1)
    except Exception:
        pass

    # Linux fallback: read /proc/meminfo when os.sysconf is unavailable
    if platform.system() == "Linux":
        try:
            from pathlib import Path

            text = Path("/proc/meminfo").read_text()
            m = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
            if m:
                return round(int(m.group(1)) / (1024**2), 1)
        except Exception:
            pass

    if platform.system() == "Windows":
        # Prefer PowerShell (Get-CimInstance) over deprecated wmic
        out = _run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
        )
        if out and out.isdigit():
            return round(int(out) / (1024**3), 1)
        # Fallback: wmic (deprecated but still common)
        out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"])
        if out:
            m = re.search(r"TotalPhysicalMemory=(\d+)", out)
            if m:
                return round(int(m.group(1)) / (1024**3), 1)
    return 0.0


def _detect_nvidia() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, mem_mb = parts[0], parts[1]
        try:
            vram_gb = round(float(mem_mb) / 1024, 1)
        except ValueError:
            continue
        gpus.append(GPUInfo(vendor="NVIDIA", name=name, vram_gb=vram_gb, source="nvidia-smi"))
    return gpus


def _detect_amd_rocm() -> list[GPUInfo]:
    if not shutil.which("rocm-smi"):
        return []
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if not out:
        return []
    gpus: list[GPUInfo] = []
    # rocm-smi CSV: header row, then data rows with device index, card name, VRAM total (bytes)
    lines = out.splitlines()
    for line in lines[1:]:  # skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[1] if parts[1] else f"AMD GPU {parts[0]}"
        try:
            vram_gb = round(float(parts[2]) / (1024**3), 1)
        except (ValueError, IndexError):
            continue
        gpus.append(GPUInfo(vendor="AMD", name=name, vram_gb=vram_gb, source="rocm-smi"))
    return gpus


def _detect_intel_gpu() -> list[GPUInfo]:
    # Try xpu-smi first (Intel's official tool for Arc/Data Center GPUs)
    if shutil.which("xpu-smi"):
        out = _run(["xpu-smi", "discovery", "--dump", "1,5"])
        if out:
            gpus: list[GPUInfo] = []
            for line in out.splitlines()[1:]:  # skip header
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                name = parts[0] if parts[0] else "Intel GPU"
                try:
                    vram_mb = float(parts[1])
                    vram_gb = round(vram_mb / 1024, 1)
                except (ValueError, IndexError):
                    continue
                gpus.append(GPUInfo(vendor="Intel", name=name, vram_gb=vram_gb, source="xpu-smi"))
            if gpus:
                return gpus

    # Fallback: parse clinfo for Intel GPUs
    if shutil.which("clinfo"):
        out = _run(["clinfo", "--raw"])
        if out:
            gpus = []
            current_name: str | None = None
            current_mem: float | None = None
            for line in out.splitlines():
                if "CL_DEVICE_NAME" in line and "intel" in line.lower():
                    current_name = line.split(None, 2)[-1].strip() if len(line.split(None, 2)) > 2 else "Intel GPU"
                elif "CL_DEVICE_GLOBAL_MEM_SIZE" in line and current_name:
                    try:
                        mem_bytes = int(line.split()[-1])
                        current_mem = round(mem_bytes / (1024**3), 1)
                    except (ValueError, IndexError):
                        pass
                if current_name and current_mem is not None:
                    gpus.append(GPUInfo(vendor="Intel", name=current_name, vram_gb=current_mem, source="clinfo"))
                    current_name = None
                    current_mem = None
            if gpus:
                return gpus
    return []


def _detect_apple_silicon() -> tuple[list[GPUInfo], float | None]:
    if platform.system() != "Darwin":
        return [], None
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    mem_out = _run(["sysctl", "-n", "hw.memsize"])
    unified_gb = round(int(mem_out) / (1024**3), 1) if mem_out and mem_out.isdigit() else None

    if chip and ("Apple" in chip or platform.machine() == "arm64"):
        approx_gpu_share = round((unified_gb or 0) * 0.65, 1) if unified_gb else 0.0
        return [
            GPUInfo(vendor="Apple", name=chip, vram_gb=approx_gpu_share, source="apple-unified-memory-estimate")
        ], unified_gb
    return [], unified_gb


def _parse_wmic_gpu_blocks(out: str) -> list[GPUInfo]:
    """Parse wmic-style Name=/AdapterRAM= output blocks into GPUInfo list."""
    blocks = [b.strip() for b in out.split("\n\n") if b.strip()]
    found: list[GPUInfo] = []
    for block in blocks:
        name_match = re.search(r"Name=(.+)", block)
        ram_match = re.search(r"AdapterRAM=(\d+)", block)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        vram_gb = 0.0
        if ram_match:
            try:
                vram_gb = round(int(ram_match.group(1)) / (1024**3), 1)
            except Exception:
                vram_gb = 0.0
        vendor = (
            "NVIDIA"
            if "nvidia" in name.lower()
            else "AMD"
            if any(x in name.lower() for x in ["amd", "radeon"])
            else "Unknown"
        )
        found.append(GPUInfo(vendor=vendor, name=name, vram_gb=vram_gb, source="wmic"))
    return found


def _detect_windows_gpu() -> list[GPUInfo]:
    if platform.system() != "Windows":
        return []

    # Prefer PowerShell (Get-CimInstance) over deprecated wmic
    ps_out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object "
            '{ "Name=$($_.Name)`nAdapterRAM=$($_.AdapterRAM)`n" }',
        ]
    )
    if ps_out:
        gpus = _parse_wmic_gpu_blocks(ps_out)
        if gpus:
            for g in gpus:
                g.source = "powershell"
            return gpus

    # Fallback: wmic (deprecated but still common)
    out = _run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:list"])
    if out:
        return _parse_wmic_gpu_blocks(out)
    return []


def _detect_cpu() -> str:
    """Return a human-readable CPU model name."""
    cpu = platform.processor() or platform.uname().processor or ""
    # platform.processor() often returns just the arch string on Linux/WSL
    if cpu in ("", "unknown", platform.machine()):
        if platform.system() == "Linux":
            try:
                from pathlib import Path

                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
            except Exception:
                pass
        if platform.system() == "Darwin":
            brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if brand:
                return brand
    return cpu or "unknown"


def detect_machine() -> MachineProfile:
    os_name = platform.system()
    arch = platform.machine()
    cpu = _detect_cpu()
    ram_gb = _detect_ram_gb()

    gpus = _detect_nvidia()
    if not gpus:
        gpus = _detect_amd_rocm()
    if not gpus:
        gpus = _detect_intel_gpu()

    apple_gpus, unified = _detect_apple_silicon()
    if not gpus and apple_gpus:
        gpus = apple_gpus

    if not gpus:
        gpus = _detect_windows_gpu()

    collapsed: dict[tuple[str, str, float, str], GPUInfo] = {}
    for g in gpus:
        key = (g.vendor, g.name, g.vram_gb, g.source)
        if key not in collapsed:
            collapsed[key] = GPUInfo(**asdict(g))
        else:
            collapsed[key].count += 1

    return MachineProfile(
        os=os_name, arch=arch, cpu=cpu, ram_gb=ram_gb, gpus=list(collapsed.values()), unified_memory_gb=unified
    )


def profile_json(profile: MachineProfile) -> str:
    return json.dumps(profile.to_dict(), indent=2)
