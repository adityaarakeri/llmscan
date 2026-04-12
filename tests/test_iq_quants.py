from __future__ import annotations

import pytest

from llmscan.vram import BITS_PER_WEIGHT, estimate_vram

# All IQ quant types that should be supported
_IQ_QUANTS = [
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL",
]

# Expected bits-per-weight ordering constraints (lower quant = fewer bits)
_BPW_ORDER = [
    ("IQ1_S", "IQ1_M"),
    ("IQ1_M", "IQ2_XXS"),
    ("IQ2_XXS", "IQ2_XS"),
    ("IQ2_XS", "IQ2_S"),
    ("IQ2_S", "IQ2_M"),
    ("IQ2_M", "IQ3_XXS"),
    ("IQ3_XXS", "IQ3_XS"),
    ("IQ3_XS", "IQ3_S"),
    ("IQ4_XS", "IQ4_NL"),
]

# Approximate expected bpw ranges based on llama.cpp documentation
_BPW_RANGES = {
    "IQ1_S":   (1.4, 1.8),
    "IQ1_M":   (1.7, 2.1),
    "IQ2_XXS": (2.0, 2.3),
    "IQ2_XS":  (2.2, 2.5),
    "IQ2_S":   (2.4, 2.7),
    "IQ2_M":   (2.6, 2.9),
    "IQ3_XXS": (2.9, 3.3),
    "IQ3_XS":  (3.2, 3.5),
    "IQ3_S":   (3.4, 3.7),
    "IQ3_M":   (3.4, 3.8),
    "IQ4_XS":  (4.1, 4.4),
    "IQ4_NL":  (4.4, 4.6),
}


class TestIqQuantsAreRegistered:
    @pytest.mark.parametrize("quant", _IQ_QUANTS)
    def test_iq_quant_in_bits_per_weight(self, quant):
        """Every IQ quant type must be present in BITS_PER_WEIGHT."""
        assert quant in BITS_PER_WEIGHT, f"{quant} missing from BITS_PER_WEIGHT"

    @pytest.mark.parametrize("quant", _IQ_QUANTS)
    def test_iq_quant_estimate_vram_does_not_raise(self, quant):
        """estimate_vram must not raise for any supported IQ quant."""
        est = estimate_vram(7, quant)
        assert est.min_vram_gb > 0

    @pytest.mark.parametrize("quant", _IQ_QUANTS)
    def test_iq_quant_case_insensitive(self, quant):
        """IQ quant lookup must be case-insensitive."""
        est_upper = estimate_vram(7, quant.upper())
        est_lower = estimate_vram(7, quant.lower())
        assert est_upper == est_lower


class TestIqQuantBitsPerWeightValues:
    @pytest.mark.parametrize("quant,bounds", list(_BPW_RANGES.items()))
    def test_bpw_in_expected_range(self, quant, bounds):
        """Bits-per-weight for each IQ quant must fall within documented range."""
        lo, hi = bounds
        bpw = BITS_PER_WEIGHT[quant]
        assert lo <= bpw <= hi, f"{quant}: bpw={bpw} not in [{lo}, {hi}]"

    @pytest.mark.parametrize("lower,higher", _BPW_ORDER)
    def test_bpw_ordering_respected(self, lower, higher):
        """Higher-quality IQ quants must have strictly more bits than lower ones."""
        assert BITS_PER_WEIGHT[lower] < BITS_PER_WEIGHT[higher], (
            f"Expected {lower} ({BITS_PER_WEIGHT[lower]}) < {higher} ({BITS_PER_WEIGHT[higher]})"
        )

    def test_all_iq_quants_below_q4_0(self):
        """IQ quants below IQ4 should use fewer bits than Q4_0 (4.5 bpw)."""
        q4_bpw = BITS_PER_WEIGHT["Q4_0"]
        for quant in ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                      "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS"]:
            assert BITS_PER_WEIGHT[quant] < q4_bpw, f"{quant} bpw should be < Q4_0"

    def test_iq4_nl_comparable_to_q4(self):
        """IQ4_NL should be in the Q4 range (≈ 4.5 bpw)."""
        bpw = BITS_PER_WEIGHT["IQ4_NL"]
        assert 4.3 <= bpw <= 4.6


class TestIqQuantVramFormula:
    def test_iq2_xs_7b_base_memory(self):
        """IQ2_XS 7B base = 7 * 2.3 / 8 = 2.0125 ≈ 2.0 GB."""
        est = estimate_vram(7, "IQ2_XS")
        assert est.base_memory_gb == pytest.approx(7 * BITS_PER_WEIGHT["IQ2_XS"] / 8, abs=0.05)

    def test_iq3_xs_7b_base_memory(self):
        """IQ3_XS 7B base = 7 * 3.3 / 8 ≈ 2.9 GB."""
        est = estimate_vram(7, "IQ3_XS")
        assert est.base_memory_gb == pytest.approx(7 * BITS_PER_WEIGHT["IQ3_XS"] / 8, abs=0.05)

    def test_iq1_s_produces_smallest_estimate(self):
        """IQ1_S must produce the smallest VRAM estimate among all IQ quants."""
        iq1s_est = estimate_vram(7, "IQ1_S")
        for quant in _IQ_QUANTS:
            if quant != "IQ1_S":
                other = estimate_vram(7, quant)
                assert iq1s_est.min_vram_gb <= other.min_vram_gb

    def test_iq4_nl_produces_largest_iq_estimate(self):
        """IQ4_NL must produce the largest VRAM estimate among all IQ quants."""
        iq4nl_est = estimate_vram(7, "IQ4_NL")
        for quant in _IQ_QUANTS:
            if quant != "IQ4_NL":
                other = estimate_vram(7, quant)
                assert iq4nl_est.min_vram_gb >= other.min_vram_gb

    def test_min_less_than_recommended_for_all_iq(self):
        """min_vram < recommended_vram for every IQ quant."""
        for quant in _IQ_QUANTS:
            est = estimate_vram(7, quant)
            assert est.min_vram_gb < est.recommended_vram_gb
