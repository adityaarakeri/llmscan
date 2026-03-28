from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CI_YAML = ROOT / ".github" / "workflows" / "ci.yml"


# ---------------------------------------------------------------------------
# CI YAML structure validation
# ---------------------------------------------------------------------------


class TestCIYaml:
    def setup_method(self) -> None:
        self.ci = yaml.safe_load(CI_YAML.read_text(encoding="utf-8"))

    def test_ci_yaml_is_valid(self):
        """CI YAML parses without error."""
        assert self.ci is not None
        assert "jobs" in self.ci

    def test_release_job_exists(self):
        assert "release" in self.ci["jobs"]

    def test_release_triggers_on_tags(self):
        """on.push.tags includes v* pattern."""
        # PyYAML parses the YAML key 'on' as boolean True
        on_config = self.ci.get("on") or self.ci.get(True)
        assert on_config is not None, "Missing 'on' trigger config"
        tags = on_config["push"].get("tags", [])
        assert any("v" in t for t in tags), f"Expected v* tag trigger, got {tags}"

    def test_release_gated_on_all_jobs(self):
        release = self.ci["jobs"]["release"]
        needs = release.get("needs", [])
        assert "lint" in needs
        assert "typecheck" in needs
        assert "test" in needs

    def test_release_has_id_token_permission(self):
        release = self.ci["jobs"]["release"]
        perms = release.get("permissions", {})
        assert perms.get("id-token") == "write"

    def test_release_has_environment(self):
        release = self.ci["jobs"]["release"]
        assert release.get("environment") == "release"

    def test_release_has_tag_condition(self):
        release = self.ci["jobs"]["release"]
        condition = release.get("if", "")
        assert "startsWith(github.ref, 'refs/tags/v')" in condition

    def test_release_uses_pypi_publish_action(self):
        release = self.ci["jobs"]["release"]
        steps = release.get("steps", [])
        action_uses = [s.get("uses", "") for s in steps]
        assert any("pypa/gh-action-pypi-publish" in u for u in action_uses), (
            f"Expected pypa/gh-action-pypi-publish action, got {action_uses}"
        )

    def test_release_builds_before_publish(self):
        release = self.ci["jobs"]["release"]
        steps = release.get("steps", [])
        run_cmds = [s.get("run", "") for s in steps]
        assert any("uv build" in cmd for cmd in run_cmds), f"Expected 'uv build' step, got {run_cmds}"


# ---------------------------------------------------------------------------
# Package build validation
# ---------------------------------------------------------------------------


class TestPackageNaming:
    def test_wheel_uses_llmscan_name(self):
        """Built wheel filename should start with llmscan, not old names."""
        import subprocess

        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", "/tmp/llmscan-test-naming"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"uv build failed: {result.stderr}"

        wheels = list(Path("/tmp/llmscan-test-naming").glob("*.whl"))
        assert len(wheels) >= 1, "No wheel produced"
        wheel_name = wheels[0].name
        assert wheel_name.startswith("llmscan-"), f"Wheel name should start with 'llmscan-', got: {wheel_name}"
        assert "llmcheck" not in wheel_name
        assert "llmfitcheck" not in wheel_name
        assert "llm_check" not in wheel_name

    def test_pyproject_package_name_is_llmscan(self):
        """pyproject.toml should declare name = 'llmscan'."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "llmscan"' in pyproject
        assert "llmcheck" not in pyproject
        assert "llmfitcheck" not in pyproject

    def test_entry_points_use_llmscan(self):
        """pyproject.toml entry points should reference llmscan module."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'llmscan = "llmscan.cli:app"' in pyproject
        assert 'llmc = "llmscan.cli:app"' in pyproject


class TestPackageBuild:
    def test_models_json_included_in_wheel(self):
        """Build the wheel and verify models.json is inside it."""
        import subprocess

        # Build in a temp location to avoid polluting the repo
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", "/tmp/llmscan-test-build"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"uv build failed: {result.stderr}"

        wheels = list(Path("/tmp/llmscan-test-build").glob("*.whl"))
        assert len(wheels) >= 1, "No wheel produced"

        with zipfile.ZipFile(wheels[0]) as zf:
            names = zf.namelist()
            assert any("models.json" in n for n in names), f"models.json not found in wheel. Contents: {names}"

    def test_build_produces_sdist_and_wheel(self):
        """uv build produces both .tar.gz and .whl."""
        import subprocess

        result = subprocess.run(
            ["uv", "build", "--out-dir", "/tmp/llmscan-test-build-full"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"uv build failed: {result.stderr}"

        out = Path("/tmp/llmscan-test-build-full")
        tarballs = list(out.glob("*.tar.gz"))
        wheels = list(out.glob("*.whl"))
        assert len(tarballs) >= 1, "No sdist (.tar.gz) produced"
        assert len(wheels) >= 1, "No wheel (.whl) produced"
