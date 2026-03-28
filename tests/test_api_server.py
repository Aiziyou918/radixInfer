import asyncio

import httpx

from radixinfer.api.server import create_app
from radixinfer.config import ServerConfig


def make_app():
    return create_app(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            device="cpu",
            start_method="inline",
            max_prefill_tokens=8,
            max_batch_size=4,
        )
    )


async def _request_json(method: str, path: str, payload: dict | None = None):
    app = make_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.request(method, path, json=payload)
            return response


async def _request_stream(path: str, payload: dict):
    app = make_app()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("POST", path, json=payload) as response:
                lines = [line async for line in response.aiter_lines() if line]
                return response, lines


def test_models_endpoint_inline_mode() -> None:
    response = asyncio.run(_request_json("GET", "/v1/models"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["id"] == "debug"


def test_nonstream_completion_inline_mode() -> None:
    response = asyncio.run(
        _request_json(
            "POST",
            "/v1/chat/completions",
            {"model": "debug", "prompt": "hi", "max_tokens": 3, "stream": False},
        )
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(payload["choices"][0]["message"]["content"], str)
    assert payload["choices"][0]["finish_reason"] in {"stop", "length"}


def test_stream_completion_inline_mode() -> None:
    response, lines = asyncio.run(
        _request_stream(
            "/v1/chat/completions",
            {"model": "debug", "prompt": "hi", "max_tokens": 3, "stream": True},
        )
    )
    assert response.status_code == 200
    body = "\n".join(lines)
    assert "chat.completion.chunk" in body
    assert "[DONE]" in body
