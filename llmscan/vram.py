from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Approximate bits per weight for common GGUF quantization types.
# Sources: llama.cpp quantization docs, empirical measurements.
BITS_PER_WEIGHT: dict[str, float] = {
    # iMatrix quantization tiers (IQ) — sorted low to high quality
    "IQ1_S":   1.6,
    "IQ1_M":   1.8,
    "IQ2_XXS": 2.1,
    "IQ2_XS":  2.3,
    "IQ2_S":   2.5,
    "IQ2_M":   2.7,
    "IQ3_XXS": 3.1,
    "IQ3_XS":  3.3,
    "IQ3_S":   3.5,
    "IQ3_M":   3.6,
    "IQ4_XS":  4.25,
    "IQ4_NL":  4.5,
    # Standard k-quants
    "Q2_K": 3.3,
    "Q3_K_S": 3.7,
    "Q3_K_M": 3.9,
    "Q3_K_L": 4.1,
    "Q4_0": 4.5,
    "Q4_K_S": 4.6,
    "Q4_K_M": 4.8,
    "Q5_0": 5.0,
    "Q5_K_S": 5.4,
    "Q5_K_M": 5.5,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "F16": 16.0,
}

# Multipliers applied to the base memory footprint.
_MIN_OVERHEAD = 1.05  # ~5% for KV cache metadata, runtime buffers
_REC_OVERHEAD = 1.3  # comfortable headroom for context window
_RAM_MULTIPLIER = 1.5  # CPU-offload / system-memory recommendation


@dataclass
class VRAMEstimate:
    """Estimated memory requirements for a quantized model."""

    base_memory_gb: float
    min_vram_gb: float
    recommended_vram_gb: float
    recommended_ram_gb: float


def estimate_vram(params_b: float, quant: str) -> VRAMEstimate:
    """Compute VRAM / RAM estimates from parameter count and quantization type.

    Raises ``ValueError`` if *quant* is not a recognised quantization type.
    """
    quant_upper = quant.upper()
    if quant_upper not in BITS_PER_WEIGHT:
        raise ValueError(f"Unknown quantization type '{quant}'. Supported types: {', '.join(sorted(BITS_PER_WEIGHT))}")

    bpw = BITS_PER_WEIGHT[quant_upper]
    base = (params_b * bpw) / 8
    return VRAMEstimate(
        base_memory_gb=round(base, 1),
        min_vram_gb=round(base * _MIN_OVERHEAD, 1),
        recommended_vram_gb=round(base * _REC_OVERHEAD, 1),
        recommended_ram_gb=round(base * _REC_OVERHEAD * _RAM_MULTIPLIER, 1),
    )


def build_model_entry(
    id: str,
    family: str,
    params_b: float,
    quant: str,
    notes: str = "",
) -> dict[str, Any]:
    """Build a complete catalog entry dict with computed VRAM/RAM values."""
    est = estimate_vram(params_b, quant)
    return {
        "id": id,
        "family": family,
        "params_b": params_b,
        "quant": quant,
        "min_vram_gb": est.min_vram_gb,
        "recommended_vram_gb": est.recommended_vram_gb,
        "recommended_ram_gb": est.recommended_ram_gb,
        "notes": notes,
    }
