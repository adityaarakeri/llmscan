from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

_MAX_CATALOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

REQUIRED_FIELDS = (
    "id",
    "family",
    "params_b",
    "quant",
    "min_vram_gb",
    "recommended_vram_gb",
    "recommended_ram_gb",
    "notes",
)


@dataclass
class ModelEntry:
    id: str
    family: str
    params_b: float
    quant: str
    min_vram_gb: float
    recommended_vram_gb: float
    recommended_ram_gb: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "params_b": self.params_b,
            "quant": self.quant,
            "min_vram_gb": self.min_vram_gb,
            "recommended_vram_gb": self.recommended_vram_gb,
            "recommended_ram_gb": self.recommended_ram_gb,
            "notes": self.notes,
        }


class CatalogValidationError(ValueError):
    """Raised when a catalog entry is missing required fields or has invalid values."""


def validate_catalog_entry(entry: dict[str, Any], index: int = 0) -> ModelEntry:
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        model_id = entry.get("id", f"entry #{index}")
        raise CatalogValidationError(f"Model '{model_id}' is missing required fields: {', '.join(missing)}")
    return ModelEntry(
        id=entry["id"],
        family=entry["family"],
        params_b=entry["params_b"],
        quant=entry["quant"],
        min_vram_gb=entry["min_vram_gb"],
        recommended_vram_gb=entry["recommended_vram_gb"],
        recommended_ram_gb=entry["recommended_ram_gb"],
        notes=entry["notes"],
    )


def validate_catalog(entries: list[dict[str, Any]]) -> list[ModelEntry]:
    return [validate_catalog_entry(e, i) for i, e in enumerate(entries)]


def _load_bundled_catalog() -> list[dict[str, Any]]:
    ref = resources.files("llmscan").joinpath("models.json")
    result: list[dict[str, Any]] = json.loads(ref.read_text(encoding="utf-8"))
    return result


DEFAULT_MODELS: list[dict[str, Any]] = _load_bundled_catalog()


def user_catalog_path() -> Path:
    """Return the path to the user's local catalog file."""
    return Path.home() / ".llmscan" / "catalog.json"


def load_user_catalog() -> list[dict[str, Any]]:
    """Load and validate the user's local catalog. Returns ``[]`` if the file does not exist."""
    path = user_catalog_path()
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size > _MAX_CATALOG_SIZE_BYTES:
            raise SystemExit(f"Error: user catalog too large ({size} bytes, limit {_MAX_CATALOG_SIZE_BYTES}): {path}")
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Error: cannot read user catalog '{path}': {exc}") from None
    try:
        entries: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"Warning: user catalog '{path}' contains invalid JSON "
            f"({exc.msg}, line {exc.lineno}). Skipping user catalog and falling back to bundled models.",
            file=sys.stderr,
        )
        return []
    try:
        validate_catalog(entries)
    except CatalogValidationError as exc:
        print(
            f"Warning: user catalog '{path}' failed validation: {exc}. "
            "Skipping user catalog and falling back to bundled models.",
            file=sys.stderr,
        )
        return []
    return entries


def save_user_catalog(entries: list[dict[str, Any]]) -> None:
    """Write entries to the user's local catalog file, creating the directory if needed."""
    path = user_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
