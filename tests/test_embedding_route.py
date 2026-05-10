from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from lens_api.api.routes import proxy
from lens_api.gateway import service
from lens_api.models import GatewayApiKey, ProtocolKind


def _test_gateway_key() -> GatewayApiKey:
    return GatewayApiKey(
        id="gateway-key-1",
        api_key="lens_test_key",
        remark="test-key",
        enabled=True,
        allowed_models=[],
        created_at="2026-05-10T00:00:00Z",
        updated_at="2026-05-10T00:00:00Z",
    )


def test_embeddings_route_accepts_post_and_strips_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_proxy_protocol(protocol, body, gateway_key):
        captured["protocol"] = protocol
        captured["body"] = body
        captured["gateway_key"] = gateway_key
        return JSONResponse(
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        )

    app = FastAPI()
    app.dependency_overrides[service.get_current_gateway_key] = _test_gateway_key
    monkeypatch.setattr(service, "_proxy_protocol", fake_proxy_protocol)
    proxy.register(app, service)

    response = TestClient(app).post(
        "/v1/embeddings",
        headers={"authorization": "Bearer test"},
        json={"model": "text-embedding-3-small", "input": "hello", "stream": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    assert captured["protocol"] is ProtocolKind.OPENAI_EMBEDDING
    assert captured["body"] == {"model": "text-embedding-3-small", "input": "hello"}
    assert isinstance(captured["gateway_key"], GatewayApiKey)


def test_embeddings_route_rejects_non_dict_body(monkeypatch) -> None:
    async def fake_proxy_protocol(protocol, body, gateway_key):
        raise AssertionError("_proxy_protocol should not be called for non-object bodies")

    app = FastAPI()
    app.dependency_overrides[service.get_current_gateway_key] = _test_gateway_key
    monkeypatch.setattr(service, "_proxy_protocol", fake_proxy_protocol)
    proxy.register(app, service)

    response = TestClient(app).post(
        "/v1/embeddings",
        headers={"authorization": "Bearer test"},
        json=["not", "an", "object"],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Embeddings request body must be a JSON object"
