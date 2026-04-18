from __future__ import annotations

import re

from typer.testing import CliRunner

from llmscan.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestCompletionFlagsExist:
    def test_install_completion_option_appears_in_help(self):
        """--install-completion is listed in the top-level --help output."""
        result = runner.invoke(app, ["--help"])
        assert "--install-completion" in _plain(result.output)

    def test_show_completion_option_appears_in_help(self):
        """--show-completion is listed in the top-level --help output."""
        result = runner.invoke(app, ["--help"])
        assert "--show-completion" in _plain(result.output)


class TestShowCompletion:
    def test_show_completion_bash_exits_zero(self):
        """--show-completion bash prints a script and exits 0."""
        result = runner.invoke(app, ["--show-completion", "bash"])
        assert result.exit_code == 0

    def test_show_completion_zsh_exits_zero(self):
        """--show-completion zsh prints a script and exits 0."""
        result = runner.invoke(app, ["--show-completion", "zsh"])
        assert result.exit_code == 0

    def test_show_completion_fish_exits_zero(self):
        """--show-completion fish prints a script and exits 0."""
        result = runner.invoke(app, ["--show-completion", "fish"])
        assert result.exit_code == 0

    def test_show_completion_bash_output_is_nonempty(self):
        """--show-completion bash produces non-empty output."""
        result = runner.invoke(app, ["--show-completion", "bash"])
        assert len(result.output.strip()) > 0

    def test_show_completion_zsh_output_is_nonempty(self):
        """--show-completion zsh produces non-empty output."""
        result = runner.invoke(app, ["--show-completion", "zsh"])
        assert len(result.output.strip()) > 0
