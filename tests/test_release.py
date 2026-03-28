from __future__ import annotations

import re
from pathlib import Path

import llmscan

ROOT = Path(__file__).resolve().parent.parent


class TestVersionSemver:
    def test_version_is_valid_semver(self):
        """__version__ must be a strict X.Y.Z semver string."""
        assert re.fullmatch(r"\d+\.\d+\.\d+", llmscan.__version__), (
            f"__version__ = {llmscan.__version__!r} is not valid semver (expected X.Y.Z)"
        )

    def test_version_parts_are_non_negative(self):
        major, minor, patch = llmscan.__version__.split(".")
        assert int(major) >= 0
        assert int(minor) >= 0
        assert int(patch) >= 0


class TestNoOldNameReferences:
    """Ensure the old package/CLI names are fully purged from source and config."""

    def test_no_llmcheck_in_source_files(self):
        source_dir = ROOT / "llmscan"
        for py_file in source_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "llmcheck" not in content, f"Old name 'llmcheck' found in {py_file.name}"
            assert "llm_check" not in content, f"Old name 'llm_check' found in {py_file.name}"

    def test_no_llmfitcheck_in_source_files(self):
        source_dir = ROOT / "llmscan"
        for py_file in source_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "llmfitcheck" not in content, f"Old name 'llmfitcheck' found in {py_file.name}"

    def test_no_old_names_in_pyproject(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "llmcheck" not in pyproject
        assert "llmfitcheck" not in pyproject
        assert "llm_check" not in pyproject

    def test_no_old_catalog_path_in_source(self):
        source_dir = ROOT / "llmscan"
        for py_file in source_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert ".llmcheck" not in content, f"Old catalog path '.llmcheck' found in {py_file.name}"


class TestVersionConsistency:
    def test_pyproject_uses_dynamic_version(self):
        """pyproject.toml should derive version from llmscan.__version__, not hardcode it."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'dynamic = ["version"]' in pyproject, "pyproject.toml should use dynamic version"
        assert 'version = {attr = "llmscan.__version__"}' in pyproject, (
            "pyproject.toml should read version from llmscan.__version__"
        )

    def test_no_hardcoded_version_in_pyproject(self):
        """pyproject.toml must NOT have a static version = 'X.Y.Z' line."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        # A static version line would look like: version = "0.3.0" (without {attr ...})
        for line in pyproject.splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped and "{" not in stripped and "dynamic" not in stripped:
                raise AssertionError(f"Found hardcoded version line in pyproject.toml: {stripped!r}")

    def test_version_not_duplicated_elsewhere(self):
        """Version string should only appear in __init__.py, not duplicated in other source files."""
        version = llmscan.__version__
        source_dir = ROOT / "llmscan"
        for py_file in source_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            # Check for hardcoded version strings like '= "0.3.0"' (assignment, not usage)
            if f'= "{version}"' in content:
                raise AssertionError(
                    f"Version {version!r} appears hardcoded in {py_file.name} - should import from __init__"
                )
