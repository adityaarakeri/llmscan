from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

STRONG = MachineProfile(
    os="Linux", arch="x86_64", cpu="i9", ram_gb=128,
    gpus=[GPUInfo(vendor="NVIDIA", name="H100", vram_gb=80.0, source="nvidia-smi")],
)

_CATALOG = [
    {"id": "model-a-7b", "family": "Alpha", "params_b": 7, "quant": "Q4_K_M",
     "min_vram_gb": 4.5, "recommended_vram_gb": 5.5, "recommended_ram_gb": 9.0, "notes": ""},
    {"id": "model-b-13b", "family": "Beta", "params_b": 13, "quant": "Q4_K_M",
     "min_vram_gb": 8.0, "recommended_vram_gb": 10.0, "recommended_ram_gb": 16.0, "notes": ""},
]


@pytest.fixture(autouse=True)
def _reset():
    cli_module._cached_profile = None
    yield
    cli_module._cached_profile = None


def _invoke(args, catalog=_CATALOG):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(catalog, f)
        name = f.name
    try:
        with patch.object(cli_module, "_get_profile", return_value=STRONG):
            result = runner.invoke(app, args + ["--catalog", name])
    finally:
        os.unlink(name)
    return result


class TestCsvFlagAccepted:
    def test_csv_flag_exits_zero(self):
        """--csv flag is accepted and exits 0."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0

    def test_csv_and_json_cannot_be_combined(self):
        """--csv and --json together produces a non-zero exit code."""
        result = _invoke(["list", "--csv", "--json"])
        assert result.exit_code != 0


class TestCsvFormat:
    def test_csv_output_has_header_row(self):
        """CSV output starts with a header row containing 'id'."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0].lower()
        assert "id" in first_line

    def test_csv_output_has_family_column(self):
        """CSV header includes 'family'."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0].lower()
        assert "family" in first_line

    def test_csv_output_has_rating_column(self):
        """CSV header includes 'rating'."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0].lower()
        assert "rating" in first_line

    def test_csv_output_is_valid_csv(self):
        """Output can be parsed as valid CSV without errors."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        assert len(rows) > 0

    def test_csv_model_ids_appear_in_output(self):
        """Each catalog model ID appears in the CSV output."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        ids = [row["id"] for row in reader]
        assert "model-a-7b" in ids
        assert "model-b-13b" in ids

    def test_csv_rating_values_are_valid(self):
        """All rating values in CSV are one of the known rating strings."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        valid_ratings = {"great", "ok", "tight", "no"}
        for row in reader:
            assert row["rating"] in valid_ratings

    def test_csv_params_column_contains_numeric_values(self):
        """The params_b column contains parseable numbers."""
        result = _invoke(["list", "--csv"])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        for row in reader:
            float(row["params_b"])  # should not raise

    def test_csv_min_rating_filter_applied(self):
        """--min-rating is respected in CSV output."""
        result = _invoke(["list", "--csv", "--min-rating", "great"])
        assert result.exit_code == 0
        reader = csv.DictReader(io.StringIO(result.output))
        rows = list(reader)
        for row in rows:
            assert row["rating"] == "great"
