from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import (
    _MAX_CATALOG_SIZE_BYTES,
    DEFAULT_MODELS,
    CatalogValidationError,
    load_user_catalog,
    validate_catalog,
)
from .detector import MachineProfile

RATING_ORDER = {"great": 4, "ok": 3, "tight": 2, "no": 1}


def load_catalog(path: str | None = None) -> list[dict[str, Any]]:
    if not path:
        merged: dict[str, dict[str, Any]] = {m["id"]: dict(m) for m in DEFAULT_MODELS}
        for m in load_user_catalog():
            merged[m["id"]] = dict(m)
        return list(merged.values())
    if not path.endswith(".json"):
        raise SystemExit(f"Error: catalog file must have a .json extension: {path}")
    # Reject paths that try to escape via symlinks or traversal to sensitive locations
    # Use PurePosixPath to ensure Unix-style paths are caught on all platforms (including Windows)
    _sensitive_prefixes = ("/etc", "/private/etc", "/proc", "/sys", "/dev", "/var/run")
    from pathlib import PurePosixPath

    posix_str = str(PurePosixPath(path))
    catalog_path = Path(path).resolve()
    resolved_str = str(catalog_path)
    if any(posix_str.startswith(p) or resolved_str.startswith(p) for p in _sensitive_prefixes):
        raise SystemExit(f"Error: catalog path is not allowed: {path}")
    try:
        size = catalog_path.stat().st_size
        if size > _MAX_CATALOG_SIZE_BYTES:
            raise SystemExit(f"Error: catalog file too large ({size} bytes): {path}")
        raw = catalog_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Error: catalog file not found: {path}") from None
    except OSError as exc:
        raise SystemExit(f"Error: cannot read catalog file '{path}': {exc}") from None
    try:
        entries: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: invalid JSON in catalog file '{path}': {exc.msg} (line {exc.lineno})") from None
    try:
        validate_catalog(entries)
    except CatalogValidationError as exc:
        raise SystemExit(f"Error: invalid catalog '{path}': {exc}") from None
    return entries


def _score_model(profile: MachineProfile, model: dict[str, Any]) -> tuple[str, str]:
    best_gpu = max((g.vram_gb for g in profile.gpus), default=0)
    total_vram = round(sum(g.vram_gb * g.count for g in profile.gpus), 1)
    system_ram = profile.ram_gb
    min_vram = float(model["min_vram_gb"])
    rec_vram = float(model["recommended_vram_gb"])
    rec_ram = float(model["recommended_ram_gb"])
    multi_gpu = len(profile.gpus) > 1 or any(g.count > 1 for g in profile.gpus)

    notes: list[str] = []

    # Primary path: single GPU meets requirements
    if best_gpu >= rec_vram and system_ram >= rec_ram:
        rating = "great"
        notes.append("GPU VRAM meets recommended target")
        notes.append("System RAM is sufficient")
    # Multi-GPU: total VRAM meets recommended, single GPU meets minimum
    elif multi_gpu and total_vram >= rec_vram and best_gpu >= min_vram and system_ram >= rec_ram:
        rating = "ok"
        notes.append(f"Total VRAM across GPUs ({total_vram} GB) meets recommended target")
        notes.append("Requires tensor parallelism or layer splitting")
    elif best_gpu >= min_vram and system_ram >= rec_ram * 0.75:
        rating = "ok"
        notes.append("GPU VRAM clears minimum target")
        notes.append("May need moderate context limits")
    # Multi-GPU: total VRAM meets minimum
    elif multi_gpu and total_vram >= min_vram and system_ram >= rec_ram * 0.75:
        rating = "tight"
        notes.append(f"Total VRAM across GPUs ({total_vram} GB) clears minimum target")
        notes.append("Requires multi-GPU setup; expect overhead")
    # CPU-only inference: no usable GPU but plenty of RAM
    elif best_gpu < min_vram * 0.3 and system_ram >= rec_ram * 1.5:
        rating = "ok"
        notes.append("Pure CPU inference viable with available RAM")
        notes.append("Expect slower tokens/sec compared to GPU")
    elif (best_gpu >= min_vram * 0.7 and system_ram >= rec_ram) or (
        best_gpu < min_vram and system_ram >= rec_ram * 1.5
    ):
        rating = "tight"
        notes.append("Likely needs CPU offload or reduced context")
        notes.append("Expect slower tokens/sec")
    else:
        rating = "no"
        notes.append("Hardware is below practical target")

    if profile.os == "Darwin" and profile.unified_memory_gb:
        notes.append("Apple Silicon estimate uses unified memory heuristics")

    return rating, "; ".join(notes)


def evaluate_models(profile: MachineProfile, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in catalog:
        rating, fit_notes = _score_model(profile, model)
        row = dict(model)
        row["rating"] = rating
        row["fit_notes"] = fit_notes
        rows.append(row)
    rows.sort(key=lambda x: (RATING_ORDER[x["rating"]], x["params_b"]), reverse=True)
    return rows
