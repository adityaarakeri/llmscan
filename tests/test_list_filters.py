from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

STRONG = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i9",
    ram_gb=128,
    gpus=[GPUInfo(vendor="NVIDIA", name="H100", vram_gb=80.0, source="nvidia-smi")],
)

_MULTI_FAMILY_CATALOG = [
    {
        "id": "llama-8b",
        "family": "Llama",
        "params_b": 8,
        "quant": "Q4_K_M",
        "min_vram_gb": 5.0,
        "recommended_vram_gb": 6.0,
        "recommended_ram_gb": 10.0,
        "notes": "",
    },
    {
        "id": "llama-70b",
        "family": "Llama",
        "params_b": 70,
        "quant": "Q4_K_M",
        "min_vram_gb": 40.0,
        "recommended_vram_gb": 50.0,
        "recommended_ram_gb": 80.0,
        "notes": "",
    },
    {
        "id": "qwen-7b",
        "family": "Qwen",
        "params_b": 7,
        "quant": "Q4_K_M",
        "min_vram_gb": 4.5,
        "recommended_vram_gb": 5.5,
        "recommended_ram_gb": 9.0,
        "notes": "",
    },
    {
        "id": "mistral-7b",
        "family": "Mistral",
        "params_b": 7,
        "quant": "Q4_K_M",
        "min_vram_gb": 4.5,
        "recommended_vram_gb": 5.5,
        "recommended_ram_gb": 9.0,
        "notes": "",
    },
]


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


def _invoke_with_catalog(args, catalog=_MULTI_FAMILY_CATALOG):
    import json as _json
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        _json.dump(catalog, f)
        name = f.name
    try:
        with patch.object(cli_module, "_get_profile", return_value=STRONG):
            result = runner.invoke(app, args + ["--catalog", name])
    finally:
        os.unlink(name)
    return result


# ---------------------------------------------------------------------------
# --family filter
# ---------------------------------------------------------------------------


class TestFamilyFilter:
    def test_family_flag_is_accepted(self):
        """--family flag is accepted without error."""
        result = _invoke_with_catalog(["list", "--family", "Llama"])
        assert result.exit_code == 0

    def test_family_filter_shows_only_matching_family(self):
        """--family Llama shows only Llama models (checked via JSON)."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "Llama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        families = {m["family"] for m in data["models"]}
        assert families == {"Llama"}

    def test_family_filter_is_case_insensitive(self):
        """--family llama (lowercase) matches Llama family entries."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "llama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [m["id"] for m in data["models"]]
        assert "llama-8b" in ids
        assert "qwen-7b" not in ids

    def test_family_filter_empty_result_exits_zero(self):
        """--family with no matches exits 0 with an empty models list."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "NonExistentFamily", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["models"] == []

    def test_family_filter_multiple_matches(self):
        """--family Llama returns all Llama entries."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "Llama", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [m["id"] for m in data["models"]]
        assert "llama-8b" in ids
        assert "llama-70b" in ids

    def test_family_filter_json_output_only_matching(self):
        """--family with --json returns only matching family in models list."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "Qwen", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [m["id"] for m in data["models"]]
        assert "qwen-7b" in ids
        assert "llama-8b" not in ids
        assert "mistral-7b" not in ids

    def test_family_filter_partial_match(self):
        """--family Lla matches Llama (substring match)."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--family", "Lla", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [m["id"] for m in data["models"]]
        assert "llama-8b" in ids
        assert "qwen-7b" not in ids


# ---------------------------------------------------------------------------
# --sort option
# ---------------------------------------------------------------------------


class TestSortOption:
    def test_sort_params_accepted(self):
        """--sort params is accepted without error."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "params"])
        assert result.exit_code == 0

    def test_sort_vram_accepted(self):
        """--sort vram is accepted without error."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "vram"])
        assert result.exit_code == 0

    def test_sort_name_accepted(self):
        """--sort name is accepted without error."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "name"])
        assert result.exit_code == 0

    def test_sort_rating_accepted(self):
        """--sort rating is the default and is accepted."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "rating"])
        assert result.exit_code == 0

    def test_sort_invalid_value_exits_nonzero(self):
        """--sort with an invalid value exits nonzero."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "invalid"])
        assert result.exit_code != 0

    def test_sort_params_descending_order(self):
        """--sort params orders models by params_b descending (largest first)."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "params", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        params = [m["params_b"] for m in data["models"]]
        assert params == sorted(params, reverse=True)

    def test_sort_vram_descending_order(self):
        """--sort vram orders models by min_vram_gb descending (most VRAM first)."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "vram", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        vrams = [m["min_vram_gb"] for m in data["models"]]
        assert vrams == sorted(vrams, reverse=True)

    def test_sort_name_ascending_order(self):
        """--sort name orders models alphabetically by id ascending."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--sort", "name", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ids = [m["id"] for m in data["models"]]
        assert ids == sorted(ids)

    def test_default_sort_is_rating(self):
        """Without --sort, models are sorted by rating descending (great first)."""
        result = _invoke_with_catalog(["list", "--min-rating", "no", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        from llmscan.estimator import RATING_ORDER

        scores = [RATING_ORDER[m["rating"]] for m in data["models"]]
        assert scores == sorted(scores, reverse=True)
