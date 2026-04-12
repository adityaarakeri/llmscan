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

# Memory overhead multipliers per backend.
# Applied to model VRAM/RAM thresholds before scoring so that backend-specific
# context-window and framework overhead are accounted for.
_BACKEND_MULTIPLIERS: dict[str, dict[str, float]] = {
    "llama-cpp": {"vram": 1.0, "ram": 1.0},  # baseline — estimates calibrated for this
    "ollama": {"vram": 1.15, "ram": 1.1},  # 15% VRAM / 10% RAM for context window + serving overhead
    "mlx": {"vram": 1.1, "ram": 1.0},  # 10% VRAM for MLX framework overhead on Apple Silicon
}
VALID_BACKENDS = frozenset(_BACKEND_MULTIPLIERS)


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


def _score_model(profile: MachineProfile, model: dict[str, Any]) -> tuple[str, str, str]:
    best_gpu = max((g.vram_gb for g in profile.gpus), default=0)
    total_vram = round(sum(g.vram_gb * g.count for g in profile.gpus), 1)
    system_ram = profile.ram_gb
    min_vram = float(model["min_vram_gb"])
    rec_vram = float(model["recommended_vram_gb"])
    rec_ram = float(model["recommended_ram_gb"])
    multi_gpu = len(profile.gpus) > 1 or any(g.count > 1 for g in profile.gpus)

    notes: list[str] = []
    reason_code: str = ""

    # Primary path: single GPU meets requirements
    if best_gpu >= rec_vram and system_ram >= rec_ram:
        rating = "great"
        reason_code = ""
        notes.append("GPU VRAM meets recommended target")
        notes.append("System RAM is sufficient")
    # Multi-GPU: total VRAM meets recommended, single GPU meets minimum
    elif multi_gpu and total_vram >= rec_vram and best_gpu >= min_vram and system_ram >= rec_ram:
        rating = "ok"
        reason_code = "multi-gpu"
        notes.append(f"Total VRAM across GPUs ({total_vram} GB) meets recommended target")
        notes.append("Requires tensor parallelism or layer splitting")
    elif best_gpu >= min_vram and system_ram >= rec_ram * 0.75:
        rating = "ok"
        if best_gpu >= rec_vram:
            reason_code = "ram low"
        else:
            deficit_pct = round((rec_vram - best_gpu) / rec_vram * 100)
            reason_code = f"vram -{deficit_pct}%"
        notes.append("GPU VRAM clears minimum target")
        notes.append("May need moderate context limits")
    # Multi-GPU: total VRAM meets minimum
    elif multi_gpu and total_vram >= min_vram and system_ram >= rec_ram * 0.75:
        rating = "tight"
        reason_code = "multi-gpu"
        notes.append(f"Total VRAM across GPUs ({total_vram} GB) clears minimum target")
        notes.append("Requires multi-GPU setup; expect overhead")
    # CPU-only inference: no usable GPU but plenty of RAM
    elif best_gpu < min_vram * 0.3 and system_ram >= rec_ram * 1.5:
        rating = "ok"
        reason_code = "cpu-only"
        notes.append("Pure CPU inference viable with available RAM")
        notes.append("Expect slower tokens/sec compared to GPU")
    elif (best_gpu >= min_vram * 0.7 and system_ram >= rec_ram) or (
        best_gpu < min_vram and system_ram >= rec_ram * 1.5
    ):
        rating = "tight"
        reason_code = "partial offload" if best_gpu >= min_vram * 0.7 and system_ram >= rec_ram else "cpu offload"
        notes.append("Likely needs CPU offload or reduced context")
        notes.append("Expect slower tokens/sec")
    else:
        rating = "no"
        reason_code = ""
        notes.append("Hardware is below practical target")

    if profile.os == "Darwin" and profile.unified_memory_gb:
        notes.append("Apple Silicon estimate uses unified memory heuristics")

    return rating, reason_code, "; ".join(notes)


def evaluate_models(
    profile: MachineProfile,
    catalog: list[dict[str, Any]],
    backend: str = "llama-cpp",
) -> list[dict[str, Any]]:
    mult = _BACKEND_MULTIPLIERS.get(backend, _BACKEND_MULTIPLIERS["llama-cpp"])
    rows: list[dict[str, Any]] = []
    for model in catalog:
        # Scale thresholds for backend overhead while keeping original values in output
        adjusted = dict(model)
        adjusted["min_vram_gb"] = round(model["min_vram_gb"] * mult["vram"], 2)
        adjusted["recommended_vram_gb"] = round(model["recommended_vram_gb"] * mult["vram"], 2)
        adjusted["recommended_ram_gb"] = round(model["recommended_ram_gb"] * mult["ram"], 2)
        rating, reason_code, fit_notes = _score_model(profile, adjusted)
        if backend != "llama-cpp":
            fit_notes = fit_notes + f"; {backend} overhead applied ({int((mult['vram'] - 1) * 100)}% VRAM)"
        row = dict(model)
        row["rating"] = rating
        row["reason_code"] = reason_code
        row["fit_notes"] = fit_notes
        rows.append(row)
    rows.sort(key=lambda x: (RATING_ORDER[x["rating"]], x["params_b"]), reverse=True)
    return rows
