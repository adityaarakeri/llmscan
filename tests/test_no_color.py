from __future__ import annotations

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import llmscan.cli as cli_module
from llmscan.cli import app
from llmscan.detector import GPUInfo, MachineProfile

runner = CliRunner()

FAKE_PROFILE = MachineProfile(
    os="Linux",
    arch="x86_64",
    cpu="Test CPU",
    ram_gb=32,
    gpus=[GPUInfo(vendor="NVIDIA", name="RTX 4090", vram_gb=24.0, source="nvidia-smi")],
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _has_ansi(text: str) -> bool:
    return bool(_ANSI_RE.search(text))


@pytest.fixture(autouse=True)
def _reset():
    """Restore the original console and profile cache after each test."""
    original_console = cli_module.console
    cli_module._cached_profile = None
    yield
    cli_module.console = original_console
    cli_module._cached_profile = None


class TestNoColorFlagAccepted:
    def test_no_color_flag_accepted_on_default_invocation(self):
        """--no-color is accepted without error on the default invocation."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color"])
        assert result.exit_code == 0

    def test_plain_flag_accepted_on_default_invocation(self):
        """--plain is accepted without error on the default invocation."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--plain"])
        assert result.exit_code == 0

    def test_no_color_flag_accepted_on_list(self):
        """--no-color is accepted before the list subcommand."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "list"])
        assert result.exit_code == 0

    def test_no_color_flag_accepted_on_scan(self):
        """--no-color is accepted before the scan subcommand."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "scan"])
        assert result.exit_code == 0

    def test_plain_flag_accepted_on_list(self):
        """--plain is accepted before the list subcommand."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--plain", "list"])
        assert result.exit_code == 0

    def test_no_color_and_json_together_accepted(self):
        """--no-color combined with --json on list works without error."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "list", "--json"])
        assert result.exit_code == 0


class TestNoColorDisablesRichColor:
    def test_no_color_sets_console_no_color_true(self):
        """After --no-color, the module-level console has no_color=True."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            runner.invoke(app, ["--no-color", "list"])
        assert cli_module.console.no_color is True

    def test_plain_sets_console_no_color_true(self):
        """After --plain, the module-level console has no_color=True."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            runner.invoke(app, ["--plain", "list"])
        assert cli_module.console.no_color is True

    def test_without_flag_console_no_color_is_false(self):
        """Without --no-color, the console no_color flag stays at its default (False)."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            runner.invoke(app, ["list"])
        assert cli_module.console.no_color is False

    def test_no_color_output_contains_no_ansi_sequences(self):
        """With --no-color and force_terminal console, output has no ANSI escape codes."""
        from rich.console import Console

        # Replace console with one that forces terminal output so Rich would normally emit ANSI
        cli_module.console = Console(force_terminal=True, no_color=False)

        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "list"])

        assert result.exit_code == 0
        assert not _has_ansi(result.output), "ANSI codes found in --no-color output"

    def test_without_no_color_forced_terminal_has_ansi(self):
        """Sanity check: a forced-terminal console without --no-color does emit ANSI codes."""
        from rich.console import Console

        cli_module.console = Console(force_terminal=True, no_color=False)

        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["list"])

        # This verifies that our forced console actually produces ANSI so the above test is meaningful
        assert _has_ansi(result.output), "Expected ANSI codes from force_terminal console but found none"


class TestNoColorSubcommands:
    def test_no_color_explain_accepted(self):
        """--no-color is accepted before the explain subcommand."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "explain", "llama-3.1-8b-instruct"])
        assert result.exit_code == 0

    def test_no_color_doctor_accepted(self):
        """--no-color is accepted before the doctor subcommand."""
        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "doctor"])
        assert result.exit_code == 0

    def test_no_color_scan_json_still_valid_json(self):
        """--no-color does not corrupt JSON output from scan --json."""
        import json

        with patch.object(cli_module, "_get_profile", return_value=FAKE_PROFILE):
            result = runner.invoke(app, ["--no-color", "scan", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)  # must not raise
        assert "os" in data
