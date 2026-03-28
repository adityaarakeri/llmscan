from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

HUGGINGFACE_API_URL = "https://huggingface.co/api"
_TIMEOUT = 15.0
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB

# Regex for recognised GGUF quant types embedded in filenames.
_QUANT_PATTERN = re.compile(
    r"(?:^|[-._])"
    r"(IQ[23]_\w+|Q[2-8]_K(?:_[SML])?|Q[2-8]_0|Q[2-8]_1|F16)"
    r"(?:[-._]|$)",
    re.IGNORECASE,
)

# Regex to extract parameter count from model names like "8B", "70b", "0.5B".
_PARAMS_PATTERN = re.compile(r"(?:^|[-._])(\d+(?:\.\d+)?)\s*[Bb](?:[-._]|$)")

# Allowed format for Hugging Face repo IDs (owner/model-name).
_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class HuggingFaceError(Exception):
    """Raised when a Hugging Face API request fails."""


@dataclass
class HFModelResult:
    repo_id: str
    author: str
    model_name: str
    downloads: int
    likes: int
    tags: list[str] = field(default_factory=list)
    last_modified: str = ""


@dataclass
class HFFileInfo:
    filename: str
    size_bytes: int


def _get_headers() -> dict[str, str]:
    """Build request headers, including auth token if available."""
    headers: dict[str, str] = {"Accept": "application/json"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _sanitize_error(exc: Exception) -> str:
    """Strip any Bearer token values from an exception message."""
    msg = str(exc)
    token = os.environ.get("HF_TOKEN")
    if token and token in msg:
        msg = msg.replace(token, "***")
    return re.sub(r"Bearer\s+[A-Za-z0-9_.-]+", "Bearer ***", msg)


def validate_repo_id(repo_id: str) -> None:
    """Validate that a repo ID matches the expected owner/name format.

    Raises ``HuggingFaceError`` if the format is invalid.
    """
    if not repo_id or not _REPO_ID_PATTERN.match(repo_id):
        raise HuggingFaceError(
            f"Invalid repo ID '{repo_id}'. Expected format: 'owner/model-name' "
            f"(alphanumeric, hyphens, underscores, and dots only)."
        )


def _read_json_response(resp: httpx.Response) -> Any:
    """Read and parse a JSON response, enforcing a size limit."""
    content_length = resp.headers.get("content-length")
    if content_length and int(content_length) > _MAX_RESPONSE_BYTES:
        raise HuggingFaceError(f"Response too large ({content_length} bytes, limit {_MAX_RESPONSE_BYTES})")
    body = resp.content
    if len(body) > _MAX_RESPONSE_BYTES:
        raise HuggingFaceError(f"Response too large ({len(body)} bytes, limit {_MAX_RESPONSE_BYTES})")
    return resp.json()


def parse_gguf_filename(filename: str) -> tuple[str, str] | None:
    """Extract ``(base_name, quant)`` from a GGUF filename.

    Returns ``None`` if the filename is not a ``.gguf`` file or contains no
    recognisable quantization pattern.
    """
    if not filename.lower().endswith(".gguf"):
        return None
    match = _QUANT_PATTERN.search(filename)
    if not match:
        return None
    quant = match.group(1).upper()
    # Derive base name: everything before the quant token, strip trailing separators.
    base = filename[: match.start()].rstrip("-._")
    return base, quant


def infer_params_from_name(name: str) -> float | None:
    """Try to extract a parameter count (in billions) from a model name string.

    Returns ``None`` if no pattern like ``8B`` or ``0.5b`` is found.
    """
    match = _PARAMS_PATTERN.search(name)
    if not match:
        return None
    return float(match.group(1))


def search_gguf_models(query: str, limit: int = 20) -> list[HFModelResult]:
    """Search Hugging Face for GGUF models matching *query*."""
    try:
        resp = httpx.get(
            f"{HUGGINGFACE_API_URL}/models",
            params={
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": limit,
            },
            headers=_get_headers(),
            timeout=_TIMEOUT,
            verify=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HuggingFaceError(f"Hugging Face search failed: {_sanitize_error(exc)}") from exc

    data = _read_json_response(resp)
    if not isinstance(data, list):
        raise HuggingFaceError("Unexpected response format from Hugging Face API")

    results: list[HFModelResult] = []
    for item in data:
        repo_id: str = item.get("modelId", item.get("id", ""))
        author = repo_id.split("/")[0] if "/" in repo_id else ""
        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        results.append(
            HFModelResult(
                repo_id=repo_id,
                author=author,
                model_name=model_name,
                downloads=item.get("downloads", 0),
                likes=item.get("likes", 0),
                tags=item.get("tags", []),
                last_modified=item.get("lastModified", ""),
            )
        )
    return results


def get_model_files(repo_id: str) -> list[HFFileInfo]:
    """Fetch the list of GGUF files available in a Hugging Face repo."""
    validate_repo_id(repo_id)
    try:
        resp = httpx.get(
            f"{HUGGINGFACE_API_URL}/models/{repo_id}",
            headers=_get_headers(),
            timeout=_TIMEOUT,
            verify=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HuggingFaceError(f"Failed to fetch model '{repo_id}': {_sanitize_error(exc)}") from exc

    data = _read_json_response(resp)
    if not isinstance(data, dict):
        raise HuggingFaceError("Unexpected response format from Hugging Face API")

    files: list[HFFileInfo] = []
    for sibling in data.get("siblings", []):
        fname = sibling.get("rfilename", "")
        if fname.lower().endswith(".gguf"):
            files.append(HFFileInfo(filename=fname, size_bytes=sibling.get("size", 0)))
    return files
