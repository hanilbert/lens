from __future__ import annotations

import httpx
import pytest

from lens_api.gateway.service import (
    _extract_model_test_text,
    _extract_response_usage,
    _extract_stream_usage,
    _extract_usage_from_payload,
    _model_test_body,
)
from lens_api.models import ProtocolKind


EXPECTED_USAGE_KEYS = {
    "resolved_model",
    "input_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "total_tokens",
}


def test_model_test_body_for_embedding_uses_input_without_stream() -> None:
    body = _model_test_body(ProtocolKind.OPENAI_EMBEDDING, "text-embedding-3-small", " hello ")

    assert body == {"model": "text-embedding-3-small", "input": "hello"}
    assert "stream" not in body


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"data": [{"embedding": [0.1, 0.2, 0.3]}]}, "<vector dim=3>"),
        ({"data": [{"embedding": "YWJjZA=="}]}, "<vector base64 len=8>"),
        ({"data": []}, ""),
        ({"data": "not-a-list"}, ""),
        ({"data": [{"object": "embedding"}]}, ""),
        ({}, ""),
    ],
)
def test_extract_model_test_text_for_embedding_payloads(
    payload: dict,
    expected: str,
) -> None:
    assert _extract_model_test_text(ProtocolKind.OPENAI_EMBEDDING, payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "text-embedding-3-small", "usage": {"prompt_tokens": 5, "total_tokens": 5}},
        {"model": "text-embedding-3-small"},
        {"model": "text-embedding-3-small", "usage": None},
        {"model": "text-embedding-3-small", "usage": "invalid"},
    ],
)
def test_extract_usage_from_payload_for_embedding_has_stable_key_set(payload: dict) -> None:
    usage = _extract_usage_from_payload(ProtocolKind.OPENAI_EMBEDDING, payload)

    assert set(usage) == EXPECTED_USAGE_KEYS


def test_extract_usage_from_payload_for_embedding_reads_prompt_tokens() -> None:
    usage = _extract_usage_from_payload(
        ProtocolKind.OPENAI_EMBEDDING,
        {"model": "text-embedding-3-small", "usage": {"prompt_tokens": 5, "total_tokens": 5}},
    )

    assert usage == {
        "resolved_model": "text-embedding-3-small",
        "input_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 5,
    }


@pytest.mark.parametrize("payload", [{}, {"usage": None}, {"usage": "invalid"}])
def test_extract_usage_from_payload_for_embedding_defaults_to_zero(payload: dict) -> None:
    usage = _extract_usage_from_payload(ProtocolKind.OPENAI_EMBEDDING, payload)

    assert usage == {
        "resolved_model": payload.get("model"),
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_extract_response_usage_for_embedding_reads_prompt_tokens() -> None:
    response = httpx.Response(
        200,
        json={"model": "text-embedding-3-small", "usage": {"prompt_tokens": 5, "total_tokens": 5}},
    )

    usage = _extract_response_usage(ProtocolKind.OPENAI_EMBEDDING, response)

    assert usage == {
        "resolved_model": "text-embedding-3-small",
        "input_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 5,
    }


@pytest.mark.parametrize("payload", [{}, {"usage": None}, {"usage": "invalid"}])
def test_extract_response_usage_for_embedding_defaults_to_zero(payload: dict) -> None:
    response = httpx.Response(200, json=payload)

    usage = _extract_response_usage(ProtocolKind.OPENAI_EMBEDDING, response)

    assert set(usage) == EXPECTED_USAGE_KEYS
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0


def test_extract_stream_usage_for_embedding_returns_zero_usage() -> None:
    usage = _extract_stream_usage(
        ProtocolKind.OPENAI_EMBEDDING,
        'data: {"usage": {"prompt_tokens": 99, "total_tokens": 99}}',
    )

    assert usage == {
        "resolved_model": None,
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
