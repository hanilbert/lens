from __future__ import annotations

import pytest
from fastapi import HTTPException

from lens_api.gateway.upstreams import build_upstream_request, resolve_channel_api_key
from lens_api.models import ChannelConfig, ChannelKeyItem, ProtocolKind


class UnsupportedProtocol:
    value = "unsupported_protocol"


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://upstream.example", "https://upstream.example/v1/embeddings"),
        ("https://upstream.example/", "https://upstream.example/v1/embeddings"),
        ("https://upstream.example/v1", "https://upstream.example/v1/embeddings"),
        ("https://upstream.example/v1/", "https://upstream.example/v1/embeddings"),
        ("https://upstream.example/v1beta", "https://upstream.example/v1/embeddings"),
        ("https://upstream.example/#", "https://upstream.example/v1/embeddings"),
    ],
)
def test_build_embedding_upstream_request_uses_embeddings_url(
    make_channel,
    settings,
    base_url: str,
    expected_url: str,
) -> None:
    upstream = build_upstream_request(
        make_channel(base_url=base_url),
        {"model": "text-embedding-3-small", "input": "hello"},
        settings,
    )

    assert upstream.method == "POST"
    assert upstream.url == expected_url


def test_build_embedding_upstream_request_sets_headers_and_merges_custom_headers(
    make_channel,
    settings,
) -> None:
    channel = make_channel(headers={"x-trace-id": "trace-1", "authorization": "Bearer override"})

    upstream = build_upstream_request(channel, {"model": "m", "input": "hello"}, settings)

    assert upstream.headers["authorization"] == "Bearer override"
    assert upstream.headers["content-type"] == "application/json"
    assert upstream.headers["x-trace-id"] == "trace-1"


def test_build_embedding_upstream_request_uses_bearer_api_key_by_default(
    make_channel,
    settings,
) -> None:
    upstream = build_upstream_request(make_channel(api_key="secret-key"), {"model": "m"}, settings)

    assert upstream.headers["authorization"] == "Bearer secret-key"


def test_build_embedding_upstream_request_copies_body_shallowly(make_channel, settings) -> None:
    body = {"model": "m", "input": "hello", "metadata": {"source": "test"}}

    upstream = build_upstream_request(make_channel(), body, settings)
    body["input"] = "changed"
    body["metadata"]["source"] = "mutated"

    assert upstream.json_body["input"] == "hello"
    assert upstream.json_body["metadata"]["source"] == "mutated"
    assert upstream.json_body is not body


def test_resolve_api_key_prefers_first_enabled_non_empty_channel_key(make_channel) -> None:
    channel = make_channel(
        api_key="fallback-key",
        keys=[
            ChannelKeyItem(id="disabled", key="disabled-key", enabled=False),
            ChannelKeyItem(id="active", key=" active-key ", enabled=True),
        ],
    )

    assert resolve_channel_api_key(channel) == "active-key"


def test_resolve_api_key_returns_first_key_when_all_channel_keys_disabled(make_channel) -> None:
    channel = make_channel(
        api_key="fallback-key",
        keys=[ChannelKeyItem(id="disabled", key=" disabled-key ", enabled=False)],
    )

    assert resolve_channel_api_key(channel) == "disabled-key"


def test_resolve_api_key_falls_back_to_channel_api_key_when_keys_empty(make_channel) -> None:
    channel = make_channel(api_key=" fallback-key ")

    assert resolve_channel_api_key(channel) == "fallback-key"


def test_resolve_api_key_can_return_empty_constructed_api_key() -> None:
    channel = ChannelConfig.model_construct(
        id="channel-empty",
        name="Empty Key Channel",
        protocol=ProtocolKind.OPENAI_EMBEDDING,
        base_url="https://upstream.example",
        api_key=" ",
        headers={},
        keys=[],
        channel_proxy="",
    )

    assert resolve_channel_api_key(channel) == ""


def test_build_upstream_request_rejects_unsupported_protocol(settings) -> None:
    channel = ChannelConfig.model_construct(
        id="channel-unsupported",
        name="Unsupported Channel",
        protocol=UnsupportedProtocol(),
        base_url="https://upstream.example",
        api_key="secret-key",
        headers={},
        keys=[],
        channel_proxy="",
    )

    with pytest.raises(HTTPException) as exc_info:
        build_upstream_request(channel, {"model": "m"}, settings)

    assert exc_info.value.status_code == 500
    assert "unsupported_protocol" in exc_info.value.detail
