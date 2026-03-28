import asyncio
from dataclasses import dataclass

import httpx

from radixinfer.api.server import AppState, create_app
from radixinfer.config import ServerConfig
from radixinfer.transport.protocol import SamplingParams, StreamChunk

# FakeAppState injects synthetic token streams without spawning any subprocesses.
# Tokens are characters from _FAKE_TEXT, so stop="b" triggers after "a".
_FAKE_TEXT = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class FakeAppState(AppState):
    def start(self) -> None:
        pass

    async def start_listener(self) -> None:
        pass

    def submit_request(
        self,
        request_id: int,
        prompt: str,
        sampling: SamplingParams,
        messages: list | None = None,
    ) -> None:
        asyncio.create_task(self._inject(request_id, sampling.max_tokens))

    async def _inject(self, request_id: int, n: int) -> None:
        # Yield once so the endpoint handler can start waiting on output_queue.get()
        await asyncio.sleep(0)
        queue = self.listeners.get(request_id)
        if queue is None:
            return
        for i in range(n):
            finished = i == n - 1
            await queue.put(
                StreamChunk(
                    request_id=request_id,
                    token_id=i + 1,
                    text=_FAKE_TEXT[i % len(_FAKE_TEXT)],
                    finished=finished,
                    finish_reason="length" if finished else "",
                    prompt_tokens=2 if finished else 0,
                    completion_tokens=n if finished else 0,
                )
            )

    async def abort_request(self, request_id: int) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def make_app():
    config = ServerConfig(
        model="debug",
        device="cpu",
        max_prefill_tokens=8,
        max_batch_size=4,
    )
    return create_app(config, state=FakeAppState(config=config))


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


def test_models_endpoint() -> None:
    response = asyncio.run(_request_json("GET", "/v1/models"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["id"] == "debug"
    assert payload["data"][0]["owned_by"] == "radixinfer"


def test_nonstream_completion() -> None:
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
    assert payload["model"] == "debug"
    assert isinstance(payload["created"], int)
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(payload["choices"][0]["message"]["content"], str)
    assert payload["choices"][0]["finish_reason"] in {"stop", "length"}
    assert payload["usage"]["prompt_tokens"] >= 1
    assert payload["usage"]["completion_tokens"] == 3
    assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"] + 3


def test_stream_completion() -> None:
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


def test_stream_completion_can_include_usage_on_final_chunk() -> None:
    response, lines = asyncio.run(
        _request_stream(
            "/v1/chat/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    )
    assert response.status_code == 200
    payloads = [line.removeprefix("data: ") for line in lines if line.startswith("data: {")]
    final = payloads[-1]
    assert '"usage"' in final
    assert '"completion_tokens": 3' in final


def test_nonstream_completion_respects_stop_sequence() -> None:
    # FakeAppState emits "a", "b", "c", ... so stop="b" truncates after "a"
    response = asyncio.run(
        _request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 6,
                "stream": False,
                "stop": "b",
            },
        )
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert "b" not in payload["choices"][0]["message"]["content"]


def test_stream_completion_respects_stop_sequence() -> None:
    response, lines = asyncio.run(
        _request_stream(
            "/v1/chat/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 6,
                "stream": True,
                "stop": "b",
            },
        )
    )
    assert response.status_code == 200
    payloads = [line.removeprefix("data: ") for line in lines if line.startswith("data: {")]
    assert any('"finish_reason": "stop"' in payload for payload in payloads)
    assert all('"content": "b"' not in payload for payload in payloads)


def test_chat_completions_rejects_n_greater_than_one() -> None:
    response = asyncio.run(
        _request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 3,
                "stream": False,
                "n": 2,
            },
        )
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only n=1 is currently supported"


def test_generate_endpoint_streams_done_marker() -> None:
    response, lines = asyncio.run(
        _request_stream(
            "/generate",
            {
                "prompt": "hi",
                "max_tokens": 3,
                "stream": True,
            },
        )
    )
    assert response.status_code == 200
    assert any(line == "data: [DONE]" for line in lines)


def test_generate_endpoint_supports_nonstream_response() -> None:
    response = asyncio.run(
        _request_json(
            "POST",
            "/generate",
            {
                "prompt": "hi",
                "max_tokens": 3,
                "stream": False,
            },
        )
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["text"], str)
    assert payload["usage"]["completion_tokens"] == 3


def test_v1_completions_nonstream_mode() -> None:
    response = asyncio.run(
        _request_json(
            "POST",
            "/v1/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 3,
                "stream": False,
            },
        )
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert isinstance(payload["choices"][0]["text"], str)
    assert payload["usage"]["completion_tokens"] == 3


def test_v1_completions_stream_mode() -> None:
    response, lines = asyncio.run(
        _request_stream(
            "/v1/completions",
            {
                "model": "debug",
                "prompt": "hi",
                "max_tokens": 3,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
    )
    assert response.status_code == 200
    body = "\n".join(lines)
    assert "text_completion.chunk" in body
    assert '"usage"' in body
    assert "[DONE]" in body
