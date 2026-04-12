from __future__ import annotations

import pytest

from llmscan.catalog import (
    DEFAULT_MODELS,
    CatalogValidationError,
    ModelEntry,
    load_user_catalog,
    save_user_catalog,
    user_catalog_path,
    validate_catalog,
    validate_catalog_entry,
)

REQUIRED_KEYS = {
    "id",
    "family",
    "params_b",
    "quant",
    "min_vram_gb",
    "recommended_vram_gb",
    "recommended_ram_gb",
    "notes",
}


# ---------------------------------------------------------------------------
# 8.13  DEFAULT_MODELS integrity
# ---------------------------------------------------------------------------


class TestDefaultModelsIntegrity:
    def test_all_entries_have_required_keys(self):
        for model in DEFAULT_MODELS:
            missing = REQUIRED_KEYS - model.keys()
            assert not missing, f"Model '{model.get('id', '???')}' missing keys: {missing}"

    def test_no_duplicate_ids(self):
        ids = [m["id"] for m in DEFAULT_MODELS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"

    def test_numeric_fields_positive(self):
        for model in DEFAULT_MODELS:
            for key in ("params_b", "min_vram_gb", "recommended_vram_gb", "recommended_ram_gb"):
                assert model[key] > 0, f"Model '{model['id']}' has non-positive {key}={model[key]}"

    def test_min_vram_lte_recommended_vram(self):
        for model in DEFAULT_MODELS:
            assert model["min_vram_gb"] <= model["recommended_vram_gb"], (
                f"Model '{model['id']}': min_vram_gb ({model['min_vram_gb']}) "
                f"> recommended_vram_gb ({model['recommended_vram_gb']})"
            )

    def test_bundled_catalog_validates(self):
        """All bundled models pass schema validation."""
        entries = validate_catalog(DEFAULT_MODELS)
        assert len(entries) == len(DEFAULT_MODELS)
        assert all(isinstance(e, ModelEntry) for e in entries)

    def test_catalog_has_at_least_45_entries(self):
        assert len(DEFAULT_MODELS) >= 45, f"Expected >= 45 models, got {len(DEFAULT_MODELS)}"

    def test_moe_models_have_moe_notes(self):
        """MoE models (Mixtral, DeepSeek V3) should mention MoE or active params."""
        moe_ids = [m for m in DEFAULT_MODELS if "mixtral" in m["id"] or "deepseek-v3" in m["id"]]
        assert len(moe_ids) >= 2, "Expected at least 2 MoE models in catalog"
        for m in moe_ids:
            notes_lower = m["notes"].lower()
            assert "moe" in notes_lower or "active" in notes_lower, (
                f"MoE model '{m['id']}' notes should mention MoE or active params: {m['notes']}"
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestCatalogValidation:
    def test_valid_entry_returns_model_entry(self):
        entry = {
            "id": "test-7b",
            "family": "Test",
            "params_b": 7,
            "quant": "Q4_K_M",
            "min_vram_gb": 5,
            "recommended_vram_gb": 8,
            "recommended_ram_gb": 16,
            "notes": "ok",
        }
        result = validate_catalog_entry(entry)
        assert isinstance(result, ModelEntry)
        assert result.id == "test-7b"

    def test_missing_field_raises(self):
        entry = {"id": "bad-model", "family": "X"}  # missing most fields
        with pytest.raises(CatalogValidationError, match="missing required fields"):
            validate_catalog_entry(entry)

    def test_missing_field_message_includes_model_id(self):
        entry = {"id": "my-model", "family": "X"}
        with pytest.raises(CatalogValidationError, match="my-model"):
            validate_catalog_entry(entry)

    def test_missing_id_uses_index_in_message(self):
        entry = {"family": "X"}  # no id at all
        with pytest.raises(CatalogValidationError, match="entry #3"):
            validate_catalog_entry(entry, index=3)

    def test_validate_catalog_all_valid(self):
        entries = [
            {
                "id": f"m{i}",
                "family": "T",
                "params_b": 7,
                "quant": "Q4",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "",
            }
            for i in range(3)
        ]
        result = validate_catalog(entries)
        assert len(result) == 3

    def test_validate_catalog_stops_on_first_bad_entry(self):
        entries = [
            {
                "id": "good",
                "family": "T",
                "params_b": 7,
                "quant": "Q4",
                "min_vram_gb": 5,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 16,
                "notes": "",
            },
            {"id": "bad"},  # missing fields
        ]
        with pytest.raises(CatalogValidationError):
            validate_catalog(entries)

    def test_model_entry_to_dict_roundtrip(self):
        entry = ModelEntry(
            id="test",
            family="T",
            params_b=7,
            quant="Q4",
            min_vram_gb=5,
            recommended_vram_gb=8,
            recommended_ram_gb=16,
            notes="note",
        )
        d = entry.to_dict()
        assert d["id"] == "test"
        assert d["min_vram_gb"] == 5
        assert set(d.keys()) == REQUIRED_KEYS


# ---------------------------------------------------------------------------
# User catalog persistence
# ---------------------------------------------------------------------------

SAMPLE_USER_ENTRY = {
    "id": "user-custom-7b",
    "family": "Custom",
    "params_b": 7,
    "quant": "Q4_K_M",
    "min_vram_gb": 4.4,
    "recommended_vram_gb": 5.5,
    "recommended_ram_gb": 8,
    "notes": "User-added model.",
}


class TestUserCatalogPath:
    def test_returns_path_ending_in_llmscan_catalog(self):
        path = user_catalog_path()
        assert path.name == "catalog.json"
        assert path.parent.name == ".llmscan"

    def test_catalog_path_does_not_use_old_name(self):
        path = str(user_catalog_path())
        assert ".llmcheck" not in path
        assert "llm_check" not in path


class TestLoadUserCatalog:
    def test_returns_empty_list_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: tmp_path / "nope" / "catalog.json")
        assert load_user_catalog() == []

    def test_returns_entries_when_valid(self, tmp_path, monkeypatch):
        import json

        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([SAMPLE_USER_ENTRY]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        result = load_user_catalog()
        assert len(result) == 1
        assert result[0]["id"] == "user-custom-7b"

    def test_returns_empty_and_warns_on_invalid_json(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "catalog.json"
        path.write_text("{bad json", encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        result = load_user_catalog()
        assert result == []
        captured = capsys.readouterr()
        assert "invalid JSON" in captured.err
        assert "Skipping user catalog" in captured.err

    def test_returns_empty_and_warns_on_missing_fields(self, tmp_path, monkeypatch, capsys):
        import json

        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([{"id": "incomplete"}]), encoding="utf-8")
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        result = load_user_catalog()
        assert result == []
        captured = capsys.readouterr()
        assert "failed validation" in captured.err
        assert "Skipping user catalog" in captured.err


class TestSaveUserCatalog:
    def test_creates_directory_and_file(self, tmp_path, monkeypatch):
        path = tmp_path / "subdir" / "catalog.json"
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        save_user_catalog([SAMPLE_USER_ENTRY])
        assert path.exists()

    def test_roundtrips_through_load(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        save_user_catalog([SAMPLE_USER_ENTRY])
        result = load_user_catalog()
        assert len(result) == 1
        assert result[0]["id"] == "user-custom-7b"

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        path = tmp_path / "catalog.json"
        monkeypatch.setattr("llmscan.catalog.user_catalog_path", lambda: path)
        save_user_catalog([SAMPLE_USER_ENTRY])
        new_entry = dict(SAMPLE_USER_ENTRY, id="replaced-model")
        save_user_catalog([new_entry])
        result = load_user_catalog()
        assert len(result) == 1
        assert result[0]["id"] == "replaced-model"
