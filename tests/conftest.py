from __future__ import annotations

import pytest

from lens_api.core.config import Settings
from lens_api.models import ChannelConfig, ChannelKeyItem, ProtocolKind


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def make_channel():
    def _make_channel(
        *,
        protocol: ProtocolKind = ProtocolKind.OPENAI_EMBEDDING,
        base_url: str = "https://upstream.example",
        api_key: str = "primary-key",
        headers: dict[str, str] | None = None,
        keys: list[ChannelKeyItem] | None = None,
    ) -> ChannelConfig:
        return ChannelConfig(
            id="channel-1",
            name="Embedding Channel",
            protocol=protocol,
            base_url=base_url,
            api_key=api_key,
            headers=headers or {},
            keys=keys or [],
        )

    return _make_channel
