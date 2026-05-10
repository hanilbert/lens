from __future__ import annotations

import pytest
from fastapi import HTTPException

from lens_api.gateway.converters import can_reach_protocol
from lens_api.gateway.upstreams import (
    _protocol_request_url,
    protocol_for_path,
    resolve_channel_base_url,
)
from lens_api.models import ProtocolKind


def test_protocol_for_embeddings_path_returns_embedding_protocol() -> None:
    assert protocol_for_path("/v1/embeddings") is ProtocolKind.OPENAI_EMBEDDING


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v1/chat/completions", ProtocolKind.OPENAI_CHAT),
        ("/v1/responses", ProtocolKind.OPENAI_RESPONSES),
        ("/v1/messages", ProtocolKind.ANTHROPIC),
        ("/v1beta/models", ProtocolKind.GEMINI),
    ],
)
def test_protocol_for_existing_paths_is_unchanged(path: str, expected: ProtocolKind) -> None:
    assert protocol_for_path(path) is expected


def test_protocol_for_unknown_path_raises_404() -> None:
    with pytest.raises(HTTPException) as exc_info:
        protocol_for_path("/v1/unknown")

    assert exc_info.value.status_code == 404


@pytest.mark.parametrize(
    ("base_url", "expected_root", "expected_url"),
    [
        ("https://api.example.com", "https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/", "https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/v1", "https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/v1/", "https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/v1beta", "https://api.example.com", "https://api.example.com/v1/embeddings"),
        ("https://api.example.com/#", "https://api.example.com", "https://api.example.com/v1/embeddings"),
    ],
)
def test_embedding_url_normalizes_base_url_variants(
    make_channel,
    base_url: str,
    expected_root: str,
    expected_url: str,
) -> None:
    channel = make_channel(base_url=base_url)

    assert resolve_channel_base_url(channel) == expected_root
    assert _protocol_request_url(channel, {"model": "text-embedding-3-small"}) == expected_url


@pytest.mark.parametrize(
    ("channel_protocol", "group_protocol", "expected"),
    [
        (ProtocolKind.OPENAI_EMBEDDING, ProtocolKind.OPENAI_EMBEDDING, True),
        (ProtocolKind.OPENAI_EMBEDDING, ProtocolKind.OPENAI_CHAT, False),
        (ProtocolKind.OPENAI_CHAT, ProtocolKind.OPENAI_EMBEDDING, False),
        (ProtocolKind.OPENAI_CHAT, ProtocolKind.OPENAI_CHAT, True),
        (ProtocolKind.OPENAI_CHAT, ProtocolKind.ANTHROPIC, True),
        (ProtocolKind.OPENAI_CHAT, ProtocolKind.OPENAI_RESPONSES, True),
        (ProtocolKind.ANTHROPIC, ProtocolKind.OPENAI_CHAT, False),
    ],
)
def test_can_reach_protocol_guards_embedding_conversions(
    channel_protocol: ProtocolKind,
    group_protocol: ProtocolKind,
    expected: bool,
) -> None:
    assert can_reach_protocol(channel_protocol, group_protocol) is expected
