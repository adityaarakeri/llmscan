from __future__ import annotations

import httpx
import pytest

from llmscan.huggingface import (
    HFFileInfo,
    HFModelResult,
    HuggingFaceError,
    get_model_files,
    infer_params_from_name,
    parse_gguf_filename,
    search_gguf_models,
)

# ---------------------------------------------------------------------------
# parse_gguf_filename
# ---------------------------------------------------------------------------


class TestParseGgufFilename:
    def test_standard_q4_k_m(self):
        result = parse_gguf_filename("model-name-Q4_K_M.gguf")
        assert result == ("model-name", "Q4_K_M")

    def test_case_insensitive(self):
        result = parse_gguf_filename("model.q5_k_m.gguf")
        assert result is not None
        assert result[1] == "Q5_K_M"

    def test_q8_0_variant(self):
        result = parse_gguf_filename("model-Q8_0-GGUF.gguf")
        assert result is not None
        assert result[1] == "Q8_0"

    def test_non_gguf_returns_none(self):
        assert parse_gguf_filename("model.safetensors") is None

    def test_gguf_no_quant_returns_none(self):
        assert parse_gguf_filename("model-latest.gguf") is None

    def test_f16_quant(self):
        result = parse_gguf_filename("llama-7b-F16.gguf")
        assert result == ("llama-7b", "F16")

    def test_iq2_xs_quant(self):
        result = parse_gguf_filename("model-IQ2_XS.gguf")
        assert result is not None
        assert result[1] == "IQ2_XS"


# ---------------------------------------------------------------------------
# infer_params_from_name
# ---------------------------------------------------------------------------


class TestInferParamsFromName:
    def test_extracts_8b(self):
        assert infer_params_from_name("llama-3.1-8b-instruct") == 8.0

    def test_extracts_70b_uppercase(self):
        assert infer_params_from_name("Qwen2.5-70B-GGUF") == 70.0

    def test_extracts_0_5b(self):
        assert infer_params_from_name("qwen2.5-0.5b") == 0.5

    def test_returns_none_no_match(self):
        assert infer_params_from_name("mistral-instruct") is None

    def test_extracts_13b(self):
        assert infer_params_from_name("codellama-13b-instruct") == 13.0


# ---------------------------------------------------------------------------
# search_gguf_models (mocked httpx)
# ---------------------------------------------------------------------------


def _mock_search_response(json_data: list[dict], status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=json_data, request=httpx.Request("GET", "https://test"))


class TestSearchGgufModels:
    def test_returns_results(self, monkeypatch):
        fake_data = [
            {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 50000,
                "likes": 200,
                "tags": ["gguf", "llama"],
                "lastModified": "2024-01-01",
            }
        ]
        monkeypatch.setattr("llmscan.huggingface.httpx.get", lambda *a, **kw: _mock_search_response(fake_data))
        results = search_gguf_models("llama")
        assert len(results) == 1
        assert isinstance(results[0], HFModelResult)
        assert results[0].repo_id == "TheBloke/Llama-2-7B-GGUF"
        assert results[0].author == "TheBloke"
        assert results[0].downloads == 50000

    def test_returns_empty_on_no_results(self, monkeypatch):
        monkeypatch.setattr("llmscan.huggingface.httpx.get", lambda *a, **kw: _mock_search_response([]))
        results = search_gguf_models("nonexistent-model-xyz")
        assert results == []

    def test_raises_on_network_error(self, monkeypatch):
        def raise_error(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("llmscan.huggingface.httpx.get", raise_error)
        with pytest.raises(HuggingFaceError, match="search failed"):
            search_gguf_models("llama")


# ---------------------------------------------------------------------------
# get_model_files (mocked httpx)
# ---------------------------------------------------------------------------


class TestGetModelFiles:
    def test_returns_gguf_files(self, monkeypatch):
        fake_data = {
            "siblings": [
                {"rfilename": "model-Q4_K_M.gguf", "size": 4_000_000_000},
                {"rfilename": "model-Q8_0.gguf", "size": 8_000_000_000},
                {"rfilename": "README.md", "size": 1000},
            ]
        }
        resp = httpx.Response(200, json=fake_data, request=httpx.Request("GET", "https://test"))
        monkeypatch.setattr("llmscan.huggingface.httpx.get", lambda *a, **kw: resp)
        files = get_model_files("TheBloke/Llama-2-7B-GGUF")
        assert len(files) == 2
        assert all(isinstance(f, HFFileInfo) for f in files)
        assert files[0].filename == "model-Q4_K_M.gguf"
        assert files[0].size_bytes == 4_000_000_000

    def test_raises_on_network_error(self, monkeypatch):
        def raise_error(*a, **kw):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("llmscan.huggingface.httpx.get", raise_error)
        with pytest.raises(HuggingFaceError, match="Failed to fetch"):
            get_model_files("some/repo")


# ---------------------------------------------------------------------------
# HF_TOKEN auth header
# ---------------------------------------------------------------------------


class TestHFTokenAuth:
    def test_token_sent_as_header(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
        captured_headers: dict[str, str] = {}

        def capture_get(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return _mock_search_response([])

        monkeypatch.setattr("llmscan.huggingface.httpx.get", capture_get)
        search_gguf_models("test")
        assert captured_headers.get("Authorization") == "Bearer hf_test_token_123"

    def test_no_auth_header_without_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        captured_headers: dict[str, str] = {}

        def capture_get(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return _mock_search_response([])

        monkeypatch.setattr("llmscan.huggingface.httpx.get", capture_get)
        search_gguf_models("test")
        assert "Authorization" not in captured_headers
