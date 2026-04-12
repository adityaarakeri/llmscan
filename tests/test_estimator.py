from __future__ import annotations

import json

import pytest

from llmscan.catalog import DEFAULT_MODELS
from llmscan.detector import GPUInfo, MachineProfile
from llmscan.estimator import RATING_ORDER, _score_model, evaluate_models, load_catalog

# ---------------------------------------------------------------------------
# 8.2  _score_model rating branches
# ---------------------------------------------------------------------------


class TestScoreModelRatings:
    """Every branch of _score_model gets a dedicated test."""

    def test_great_single_gpu_meets_recommended(self, great_profile, sample_model):
        rating, _, notes = _score_model(great_profile, sample_model)
        assert rating == "great"
        assert "GPU VRAM meets recommended target" in notes
        assert "System RAM is sufficient" in notes

    def test_ok_single_gpu_meets_minimum(self, sample_model):
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=14,
            gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=6.0, source="nvidia-smi")],
        )
        rating, _, notes = _score_model(profile, sample_model)
        assert rating == "ok"
        assert "GPU VRAM clears minimum target" in notes

    def test_ok_multi_gpu_total_meets_recommended(self, sample_model):
        # Two GPUs: 4 GB each = 8 GB total >= rec 8, each 4 < rec 8 but >= min 5.5? No.
        # Need each >= min_vram (5.5). Use 2x 6 GB = 12 GB total.
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="Xeon",
            ram_gb=32,
            gpus=[
                GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=6.0, source="nvidia-smi"),
                GPUInfo(vendor="NVIDIA", name="RTX 3060 Ti", vram_gb=6.0, source="nvidia-smi"),
            ],
        )
        rating, _, notes = _score_model(profile, sample_model)
        assert rating == "ok"
        assert "Total VRAM across GPUs" in notes
        assert "tensor parallelism" in notes

    def test_tight_multi_gpu_total_meets_minimum(self, sample_model):
        # Two GPUs: 3 GB each = 6 GB total >= min 5.5 but < rec 8, each < min 5.5
        # best_gpu=3.0 < min 5.5, so single-GPU ok branch won't fire.
        # multi_gpu=True, total_vram=6.0 >= min 5.5, ram >= 0.75*16=12
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="Xeon",
            ram_gb=16,
            gpus=[
                GPUInfo(vendor="NVIDIA", name="GTX 1060", vram_gb=3.0, source="nvidia-smi"),
                GPUInfo(vendor="NVIDIA", name="GTX 1060", vram_gb=3.0, source="nvidia-smi"),
            ],
        )
        rating, _, notes = _score_model(profile, sample_model)
        assert rating == "tight"
        assert "Total VRAM across GPUs" in notes
        assert "multi-GPU" in notes

    def test_ok_cpu_only_high_ram(self, sample_model):
        # 0 VRAM, RAM >= 1.5 * 16 = 24, and best_gpu < min * 0.3 = 1.65
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i9",
            ram_gb=64,
            gpus=[],
        )
        rating, _, notes = _score_model(profile, sample_model)
        assert rating == "ok"
        assert "Pure CPU inference" in notes

    def test_tight_partial_offload(self, sample_model):
        # GPU >= 70% of min (5.5*0.7=3.85), RAM >= rec (16)
        # But GPU < min (5.5) so single-GPU ok won't fire.
        # best_gpu=4.0 >= 3.85 and ram=16 >= 16 → tight via partial offload
        # Also need to make sure multi-gpu branches don't fire (single GPU).
        # And CPU-only won't fire because best_gpu (4.0) >= min*0.3 (1.65).
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=16,
            gpus=[GPUInfo(vendor="NVIDIA", name="GTX 1650", vram_gb=4.0, source="nvidia-smi")],
        )
        rating, _, notes = _score_model(profile, sample_model)
        assert rating == "tight"
        assert "offload" in notes.lower() or "slower" in notes.lower()

    def test_tight_low_gpu_high_ram(self, sample_model):
        # GPU < min but RAM >= 1.5x rec. GPU >= min*0.3 so CPU-only won't fire.
        # best_gpu=2.0 >= min*0.3=1.65, so cpu-only (needs <1.65) won't fire.
        # best_gpu=2.0 < min*0.7=3.85, so partial offload (first condition) won't fire.
        # But second condition: best_gpu < min AND ram >= rec*1.5 → tight.
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=32,
            gpus=[GPUInfo(vendor="NVIDIA", name="GT 1030", vram_gb=2.0, source="nvidia-smi")],
        )
        rating, _, _ = _score_model(profile, sample_model)
        assert rating == "tight"

    def test_no_insufficient_hardware(self, weak_profile, sample_model):
        rating, _, notes = _score_model(weak_profile, sample_model)
        assert rating == "no"
        assert "below practical target" in notes

    def test_apple_silicon_note(self, apple_profile, sample_model):
        _, _, notes = _score_model(apple_profile, sample_model)
        assert "unified memory heuristics" in notes

    def test_boundary_exact_recommended(self, sample_model):
        """GPU == rec_vram exactly, RAM == rec_ram exactly → great."""
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=16.0,
            gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3070", vram_gb=8.0, source="nvidia-smi")],
        )
        rating, _, _ = _score_model(profile, sample_model)
        assert rating == "great"

    def test_boundary_just_below_recommended(self, sample_model):
        """GPU just under rec_vram, at min → ok."""
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=16.0,
            gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=7.9, source="nvidia-smi")],
        )
        rating, _, _ = _score_model(profile, sample_model)
        assert rating == "ok"

    def test_boundary_exact_minimum(self, sample_model):
        """GPU == min_vram exactly, RAM at 75% rec → ok."""
        profile = MachineProfile(
            os="Linux",
            arch="x86_64",
            cpu="i7",
            ram_gb=12.0,
            gpus=[GPUInfo(vendor="NVIDIA", name="RTX 3060", vram_gb=5.5, source="nvidia-smi")],
        )
        rating, _, _ = _score_model(profile, sample_model)
        assert rating == "ok"


# ---------------------------------------------------------------------------
# 8.3  evaluate_models
# ---------------------------------------------------------------------------


class TestEvaluateModels:
    def test_sort_order_rating_then_params(self, great_profile):
        catalog = [
            {
                "id": "small",
                "family": "T",
                "params_b": 3,
                "quant": "Q4",
                "min_vram_gb": 2,
                "recommended_vram_gb": 4,
                "recommended_ram_gb": 8,
                "notes": "",
            },
            {
                "id": "big",
                "family": "T",
                "params_b": 70,
                "quant": "Q4",
                "min_vram_gb": 42,
                "recommended_vram_gb": 48,
                "recommended_ram_gb": 64,
                "notes": "",
            },
        ]
        rows = evaluate_models(great_profile, catalog)
        # "small" should be great (24 GB >= 4, 32 >= 8). "big" should be no.
        assert rows[0]["id"] == "small"
        assert rows[0]["rating"] == "great"
        assert rows[-1]["id"] == "big"

    def test_all_catalog_entries_scored(self, great_profile):
        catalog = load_catalog(None)
        rows = evaluate_models(great_profile, catalog)
        assert len(rows) == len(catalog)

    def test_rows_have_rating_and_fit_notes(self, great_profile, sample_model):
        rows = evaluate_models(great_profile, [sample_model])
        assert "rating" in rows[0]
        assert "fit_notes" in rows[0]
        assert "reason_code" in rows[0]
        assert rows[0]["rating"] in RATING_ORDER


# ---------------------------------------------------------------------------
# 8.4  load_catalog
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_default_catalog_length(self):
        catalog = load_catalog(None)
        assert len(catalog) == len(DEFAULT_MODELS)

    def test_defensive_copy(self):
        catalog = load_catalog(None)
        catalog.append({"id": "injected"})
        assert len(DEFAULT_MODELS) != len(catalog)
        # Also verify dict-level copy
        catalog2 = load_catalog(None)
        catalog2[0]["injected_key"] = True
        assert "injected_key" not in DEFAULT_MODELS[0]

    def test_custom_catalog_from_file(self, tmp_path):
        data = [
            {
                "id": "custom-model",
                "family": "X",
                "params_b": 7,
                "quant": "Q4",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "custom",
            }
        ]
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_catalog(str(path))
        assert len(result) == 1
        assert result[0]["id"] == "custom-model"

    def test_missing_file_raises_system_exit(self):
        with pytest.raises(SystemExit, match="catalog file not found"):
            load_catalog("/nonexistent/path/catalog.json")

    def test_malformed_json_raises_system_exit(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SystemExit, match="invalid JSON"):
            load_catalog(str(path))

    def test_custom_catalog_missing_fields_raises_system_exit(self, tmp_path):
        data = [{"id": "incomplete-model", "family": "X"}]  # missing required fields
        path = tmp_path / "bad_catalog.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(SystemExit, match="invalid catalog"):
            load_catalog(str(path))


# ---------------------------------------------------------------------------
# User catalog merge behavior
# ---------------------------------------------------------------------------

_USER_ENTRY = {
    "id": "user-added-7b",
    "family": "Custom",
    "params_b": 7,
    "quant": "Q4_K_M",
    "min_vram_gb": 4.4,
    "recommended_vram_gb": 5.5,
    "recommended_ram_gb": 8,
    "notes": "From user catalog.",
}


class TestLoadCatalogMerge:
    def test_merges_user_catalog(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([_USER_ENTRY]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        catalog = load_catalog(None)
        ids = [m["id"] for m in catalog]
        assert "user-added-7b" in ids
        assert len(catalog) == len(DEFAULT_MODELS) + 1

    def test_user_entry_overrides_bundled(self, tmp_path, monkeypatch):
        override = dict(DEFAULT_MODELS[0], notes="USER OVERRIDE")
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([override]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        catalog = load_catalog(None)
        matched = [m for m in catalog if m["id"] == override["id"]]
        assert len(matched) == 1
        assert matched[0]["notes"] == "USER OVERRIDE"
        # Total count unchanged since it's an override, not addition
        assert len(catalog) == len(DEFAULT_MODELS)

    def test_explicit_catalog_ignores_user_catalog(self, tmp_path, monkeypatch):
        user_path = tmp_path / "user_catalog.json"
        user_path.write_text(json.dumps([_USER_ENTRY]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: user_path)

        custom_entry = dict(_USER_ENTRY, id="custom-only")
        custom_path = tmp_path / "custom.json"
        custom_path.write_text(json.dumps([custom_entry]), encoding="utf-8")

        catalog = load_catalog(str(custom_path))
        ids = [m["id"] for m in catalog]
        assert "custom-only" in ids
        assert "user-added-7b" not in ids
        assert len(catalog) == 1

    def test_defensive_copy_with_user_catalog(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([_USER_ENTRY]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        catalog = load_catalog(None)
        catalog[0]["injected_key"] = True
        assert "injected_key" not in DEFAULT_MODELS[0]
