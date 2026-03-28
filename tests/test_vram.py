from __future__ import annotations

import pytest

from llmscan.catalog import REQUIRED_FIELDS, validate_catalog_entry
from llmscan.vram import BITS_PER_WEIGHT, VRAMEstimate, build_model_entry, estimate_vram


class TestEstimateVram:
    """Tests for the core VRAM estimation formula."""

    def test_known_q4_k_m_70b(self):
        """70B Q4_K_M should produce ~42 GB base (matches hand-curated catalog)."""
        est = estimate_vram(70, "Q4_K_M")
        assert est.base_memory_gb == 42.0
        assert est.min_vram_gb == 44.1
        assert est.recommended_vram_gb == 54.6
        assert est.recommended_ram_gb == 81.9

    def test_known_q4_k_m_8b(self):
        """8B Q4_K_M base should be ~4.8 GB."""
        est = estimate_vram(8, "Q4_K_M")
        assert est.base_memory_gb == 4.8

    def test_f16_7b(self):
        """F16 7B should be 14 GB base (7 * 16 / 8)."""
        est = estimate_vram(7, "F16")
        assert est.base_memory_gb == 14.0

    def test_q8_0_3b(self):
        """Q8_0 3B should be ~3.2 GB base."""
        est = estimate_vram(3, "Q8_0")
        assert est.base_memory_gb == 3.2

    def test_min_less_than_recommended(self):
        """min_vram should always be less than recommended_vram."""
        for quant in BITS_PER_WEIGHT:
            est = estimate_vram(7, quant)
            assert est.min_vram_gb <= est.recommended_vram_gb

    def test_recommended_vram_less_than_ram(self):
        """recommended_vram should always be less than recommended_ram."""
        for quant in BITS_PER_WEIGHT:
            est = estimate_vram(14, quant)
            assert est.recommended_vram_gb <= est.recommended_ram_gb

    def test_case_insensitive_quant(self):
        """Quant strings should be case-insensitive."""
        lower = estimate_vram(7, "q4_k_m")
        upper = estimate_vram(7, "Q4_K_M")
        assert lower == upper

    def test_unknown_quant_raises(self):
        """An unknown quant type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization type"):
            estimate_vram(7, "Q99_FAKE")

    def test_returns_vram_estimate_dataclass(self):
        est = estimate_vram(7, "Q4_K_M")
        assert isinstance(est, VRAMEstimate)


class TestBuildModelEntry:
    """Tests for the catalog entry builder."""

    def test_has_all_required_fields(self):
        entry = build_model_entry("test-7b", "Test", 7, "Q4_K_M", "A test model.")
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing required field: {field}"

    def test_passes_catalog_validation(self):
        entry = build_model_entry("test-7b", "Test", 7, "Q4_K_M", "A test model.")
        model = validate_catalog_entry(entry)
        assert model.id == "test-7b"
        assert model.family == "Test"
        assert model.params_b == 7

    def test_preserves_inputs(self):
        entry = build_model_entry("my-model", "MyFamily", 14, "Q5_K_M", "Some notes")
        assert entry["id"] == "my-model"
        assert entry["family"] == "MyFamily"
        assert entry["params_b"] == 14
        assert entry["quant"] == "Q5_K_M"
        assert entry["notes"] == "Some notes"

    def test_computed_values_match_estimate(self):
        entry = build_model_entry("x", "X", 32, "Q4_K_M")
        est = estimate_vram(32, "Q4_K_M")
        assert entry["min_vram_gb"] == est.min_vram_gb
        assert entry["recommended_vram_gb"] == est.recommended_vram_gb
        assert entry["recommended_ram_gb"] == est.recommended_ram_gb

    def test_default_empty_notes(self):
        entry = build_model_entry("x", "X", 7, "Q4_K_M")
        assert entry["notes"] == ""
