from __future__ import annotations

import pytest

from llmscan.detector import GPUInfo, MachineProfile


@pytest.fixture
def sample_model():
    """A representative 8B model entry for testing."""
    return {
        "id": "test-model-8b",
        "family": "TestFamily",
        "params_b": 8,
        "quant": "Q4_K_M",
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 8,
        "recommended_ram_gb": 16,
        "notes": "Test model.",
    }


@pytest.fixture
def great_profile():
    """Machine that exceeds recommended specs for an 8B model."""
    return MachineProfile(
        os="Linux",
        arch="x86_64",
        cpu="Intel i9",
        ram_gb=32,
        gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
    )


@pytest.fixture
def no_gpu_profile():
    """Machine with no GPU at all."""
    return MachineProfile(
        os="Linux",
        arch="x86_64",
        cpu="Intel i7",
        ram_gb=64,
        gpus=[],
    )


@pytest.fixture
def dual_gpu_profile():
    """Machine with two identical GPUs."""
    return MachineProfile(
        os="Linux",
        arch="x86_64",
        cpu="Intel Xeon",
        ram_gb=64,
        gpus=[
            GPUInfo(vendor="NVIDIA", name="RTX 3090", vram_gb=24.0, count=2, source="nvidia-smi"),
        ],
    )


@pytest.fixture
def apple_profile():
    """Apple Silicon machine with unified memory."""
    return MachineProfile(
        os="Darwin",
        arch="arm64",
        cpu="Apple M2 Max",
        ram_gb=32,
        gpus=[GPUInfo(vendor="Apple", name="Apple M2 Max", vram_gb=20.8, source="apple-unified-memory-estimate")],
        unified_memory_gb=32,
    )


@pytest.fixture
def weak_profile():
    """Machine well below requirements."""
    return MachineProfile(
        os="Linux",
        arch="x86_64",
        cpu="Celeron",
        ram_gb=4,
        gpus=[GPUInfo(vendor="NVIDIA", name="GT 730", vram_gb=1.0, source="nvidia-smi")],
    )
