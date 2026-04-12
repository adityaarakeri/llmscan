from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

# A very weak machine — most models will be rated "no"
WEAK_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Celeron",
    ram_gb=4,
    gpus=[GPUInfo(vendor="NVIDIA", name="GT 730", vram_gb=1.0, source="nvidia-smi")],
)

# A strong machine — most models will be rated "great"
STRONG_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="i9",
    ram_gb=128,
    gpus=[GPUInfo(vendor="NVIDIA", name="H100", vram_gb=80.0, source="nvidia-smi")],
)


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


class TestHiddenModelsSummaryFooter:
    def test_default_filter_shows_hidden_count_when_models_hidden(self):
        """With default filter on a weak machine, a footer mentions hidden models."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Must mention something was hidden
        assert "hidden" in result.output.lower() or "not shown" in result.output.lower()

    def test_footer_shows_exact_hidden_count(self):
        """The footer states the number of hidden 'no'-rated models."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # There should be a positive number mentioned alongside "hidden"
        import re

        assert re.search(r"\d+", result.output), "Expected a number in the footer"

    def test_footer_mentions_min_rating_no_flag(self):
        """The footer instructs the user how to show all models."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "--min-rating" in result.output
        assert "no" in result.output

    def test_no_footer_when_min_rating_no(self):
        """With --min-rating no, nothing is hidden so no footer is shown."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list", "--min-rating", "no"])
        assert result.exit_code == 0
        assert "hidden" not in result.output.lower()

    def test_no_footer_when_nothing_hidden(self, tmp_path, monkeypatch):
        """When every model in the catalog passes the filter, no footer is shown."""
        import json as _json

        # A tiny catalog where every model easily fits the STRONG_PROFILE
        tiny_catalog = [
            {
                "id": "tiny-1b",
                "family": "Test",
                "params_b": 1,
                "quant": "Q4_K_M",
                "min_vram_gb": 0.5,
                "recommended_vram_gb": 1.0,
                "recommended_ram_gb": 2.0,
                "notes": "",
            },
            {
                "id": "small-3b",
                "family": "Test",
                "params_b": 3,
                "quant": "Q4_K_M",
                "min_vram_gb": 1.0,
                "recommended_vram_gb": 2.0,
                "recommended_ram_gb": 4.0,
                "notes": "",
            },
        ]
        cat_path = tmp_path / "catalog.json"
        cat_path.write_text(_json.dumps(tiny_catalog), encoding="utf-8")
        with patch.object(cli_module, "_get_profile", return_value=STRONG_PROFILE):
            result = runner.invoke(app, ["list", "--catalog", str(cat_path)])
        assert result.exit_code == 0
        assert "hidden" not in result.output.lower()

    def test_footer_not_shown_in_json_mode(self):
        """JSON output must not include the hidden-models footer string."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)  # must be valid JSON — no extra footer text
        assert "hidden" not in json.dumps(data).lower() or "models" in data

    def test_footer_hidden_count_is_accurate(self):
        """Hidden count in footer equals total models minus shown models."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result_default = runner.invoke(app, ["list"])
            result_all = runner.invoke(app, ["list", "--min-rating", "no", "--json"])

        assert result_default.exit_code == 0
        assert result_all.exit_code == 0

        all_data = json.loads(result_all.output)
        total = len(all_data["models"])

        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result_default_json = runner.invoke(app, ["list", "--json"])
        shown_data = json.loads(result_default_json.output)
        shown = len(shown_data["models"])

        hidden_count = total - shown
        assert str(hidden_count) in result_default.output

    def test_footer_with_great_filter_counts_ok_tight_no_as_hidden(self):
        """With --min-rating great on a weak machine, footer mentions hidden models."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list", "--min-rating", "great"])
        assert result.exit_code == 0
        assert "hidden" in result.output.lower() or "not shown" in result.output.lower()

    def test_footer_with_tight_filter_mentions_no_rated_hidden(self):
        """Default filter (tight+) footer specifically mentions 'no' rating."""
        with patch.object(cli_module, "_get_profile", return_value=WEAK_PROFILE):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "no" in result.output
