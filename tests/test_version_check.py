from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from llmscan import __version__
from llmscan.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _mock_pypi(version: str, status: int = 200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = {"info": {"version": version}}
    m.raise_for_status = MagicMock()
    return m


class TestVersionCommandExists:
    def test_version_command_exits_zero(self):
        """'llmscan version' subcommand exits 0."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_command_prints_current_version(self):
        """'llmscan version' prints the installed version string."""
        result = runner.invoke(app, ["version"])
        assert __version__ in result.output

    def test_version_check_flag_exists(self):
        """'llmscan version --help' lists the --check flag."""
        result = runner.invoke(app, ["version", "--help"])
        assert "--check" in _plain(result.output)


class TestVersionCheckUpToDate:
    def test_up_to_date_message_when_versions_match(self):
        """When local == PyPI version, output says the tool is up to date."""
        with patch("httpx.get", return_value=_mock_pypi(__version__)):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0
        assert "up to date" in result.output.lower() or __version__ in result.output

    def test_up_to_date_exits_zero(self):
        """When up to date, exit code is 0."""
        with patch("httpx.get", return_value=_mock_pypi(__version__)):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0


class TestVersionCheckOutdated:
    def test_update_available_message_when_newer_version_on_pypi(self):
        """When PyPI has a newer version, output mentions an update is available."""
        with patch("httpx.get", return_value=_mock_pypi("99.99.99")):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0
        output = result.output.lower()
        assert "update" in output or "99.99.99" in result.output

    def test_new_version_number_shown_when_outdated(self):
        """The newer PyPI version number is shown when an update is available."""
        with patch("httpx.get", return_value=_mock_pypi("99.99.99")):
            result = runner.invoke(app, ["version", "--check"])
        assert "99.99.99" in result.output

    def test_outdated_exits_zero(self):
        """Outdated version check still exits 0 — it is informational only."""
        with patch("httpx.get", return_value=_mock_pypi("99.99.99")):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0


class TestVersionCheckNetworkFailure:
    def test_network_error_exits_zero(self):
        """A network failure during --check exits 0 with a warning, not a crash."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0

    def test_network_error_shows_warning_not_traceback(self):
        """Network failure during --check prints a human-friendly warning."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = runner.invoke(app, ["version", "--check"])
        assert "Traceback" not in result.output
        assert "pypi" in result.output.lower() or "check" in result.output.lower() or "error" in result.output.lower()

    def test_timeout_is_handled_gracefully(self):
        """A timeout during --check is handled without crashing."""
        import httpx

        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            result = runner.invoke(app, ["version", "--check"])
        assert result.exit_code == 0
