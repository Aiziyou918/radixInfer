from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Callable

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from fastapi.exceptions import HTTPException
    from pydantic import BaseModel
except ModuleNotFoundError:
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

    class BaseModel:  # type: ignore[no-redef]
        pass

from radixinfer.config import ServerConfig
from radixinfer.server.common import build_usage
from radixinfer.transport.protocol import (
    SamplingParams,
    StreamChunk,
)
from radixinfer.server.frontend import FrontendManager


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False
    ignore_eos: bool = False
    stream_options: StreamOptions | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    stream: bool = True
    ignore_eos: bool = False
    stop: str | list[str] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = True
    ignore_eos: bool = False
    stream_options: StreamOptions | None = None


AppState = FrontendManager


def _flatten_prompt(request: ChatCompletionRequest) -> str:
    if request.messages:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
    return request.prompt or ""


def _normalize_stop_sequences(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,) if stop else ()
    return tuple(item for item in stop if item)


def _truncate_for_stop_sequences(text: str, stop_sequences: tuple[str, ...]) -> tuple[str, bool]:
    stop_idx: int | None = None
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx >= 0 and (stop_idx is None or idx < stop_idx):
            stop_idx = idx
    if stop_idx is None:
        return text, False
    return text[:stop_idx], True


class _StreamingStopState:
    def __init__(self, stop_sequences: tuple[str, ...]) -> None:
        self.stop_sequences = stop_sequences
        self._pending = ""
        self._tail_len = max((len(stop) for stop in stop_sequences), default=0) - 1

    def push(self, text: str) -> tuple[str, bool]:
        if not self.stop_sequences:
            return text, False
        combined = self._pending + text
        truncated, matched = _truncate_for_stop_sequences(combined, self.stop_sequences)
        if matched:
            self._pending = ""
            return truncated, True
        if self._tail_len <= 0:
            self._pending = ""
            return combined, False
        split_at = max(0, len(combined) - self._tail_len)
        self._pending = combined[split_at:]
        return combined[:split_at], False

    def flush(self) -> str:
        remaining = self._pending
        self._pending = ""
        return remaining


def _build_stream_payload(
    request_id: int,
    model: str,
    created: int,
    delta: dict,
    finish_reason: str | None,
    usage: dict[str, int] | None = None,
) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        **({"usage": usage} if usage is not None else {}),
    }


def _build_completion_payload(
    request_id: int,
    model: str,
    created: int,
    content: str,
    finish_reason: str,
    usage: dict[str, int],
) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _build_text_completion_payload(
    request_id: int,
    model: str,
    created: int,
    content: str,
    finish_reason: str,
    usage: dict[str, int],
) -> dict:
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _build_text_stream_payload(
    request_id: int,
    model: str,
    created: int,
    text: str,
    finish_reason: str | None,
    usage: dict[str, int] | None = None,
) -> dict:
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
        **({"usage": usage} if usage is not None else {}),
    }


def _make_sampling_params(payload, stop_sequences: tuple[str, ...]) -> SamplingParams:
    return SamplingParams(
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        ignore_eos=payload.ignore_eos,
        stop=stop_sequences,
        n=getattr(payload, "n", 1),
        presence_penalty=getattr(payload, "presence_penalty", 0.0),
        frequency_penalty=getattr(payload, "frequency_penalty", 0.0),
    )


def _should_include_usage(stream_options: StreamOptions | None, *, finished: bool) -> bool:
    return finished and stream_options is not None and stream_options.include_usage


def _sse_text_frame(text: str) -> str:
    return f"data: {text}\n"


def _sse_json_frame(body: dict) -> str:
    return f"data: {json.dumps(body)}\n\n"


async def _stream_with_stop_handling(
    *,
    state: FrontendManager,
    request: Request,
    request_id: int,
    output_queue: asyncio.Queue[StreamChunk],
    stop_sequences: tuple[str, ...],
    render_chunk: Callable[[StreamChunk, str, str | None], str | None],
    render_tail: Callable[[str], str | None],
    done_frame: str,
):
    matcher = _StreamingStopState(stop_sequences)
    try:
        while True:
            if await request.is_disconnected():
                await state.abort_request(request_id)
                break
            chunk = await output_queue.get()
            text = chunk.text
            matched_stop = False
            if text:
                text, matched_stop = matcher.push(text)
            finish_reason = None if not chunk.finished else chunk.finish_reason
            if matched_stop:
                finish_reason = "stop"
            frame = render_chunk(chunk, text, finish_reason)
            if frame is not None:
                yield frame
            if matched_stop:
                await state.abort_request(request_id)
                break
            if chunk.finished:
                tail = matcher.flush()
                if tail:
                    tail_frame = render_tail(tail)
                    if tail_frame is not None:
                        yield tail_frame
                break
        yield done_frame
    finally:
        state.close_listener(request_id)


async def _collect_truncated_response(
    state: FrontendManager,
    request_id: int,
    output_queue: asyncio.Queue[StreamChunk],
    stop_sequences: tuple[str, ...],
) -> tuple[str, str, dict[str, int]]:
    content, finish_reason, usage = await state.collect_response(request_id, output_queue)
    content, matched_stop = _truncate_for_stop_sequences(content, stop_sequences)
    if matched_stop:
        finish_reason = "stop"
    return content, finish_reason, usage


def create_app(
    config: ServerConfig,
    state: FrontendManager | None = None,
    *,
    manage_backend: bool = True,
) -> FastAPI:
    if FastAPI is None or Request is None or StreamingResponse is None:
        raise RuntimeError("FastAPI dependencies are not installed. Install project dependencies first.")
    if state is None:
        state = AppState(config=config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if manage_backend:
            state.start()
        await state.start_listener()
        yield
        await state.shutdown()

    app = FastAPI(title="radixInfer", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "radixinfer",
                    "root": config.model,
                }
            ],
        }

    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(payload: GenerateRequest, request: Request):
        stop_sequences = _normalize_stop_sequences(payload.stop)
        request_id, output_queue = state.open_request(
            payload.prompt,
            _make_sampling_params(payload, stop_sequences),
        )

        async def stream():
            async for frame in _stream_with_stop_handling(
                state=state,
                request=request,
                request_id=request_id,
                output_queue=output_queue,
                stop_sequences=stop_sequences,
                render_chunk=lambda _chunk, text, _finish_reason: (
                    _sse_text_frame(text) if text else None
                ),
                render_tail=lambda tail: _sse_text_frame(tail),
                done_frame="data: [DONE]\n",
            ):
                yield frame

        if not payload.stream:
            try:
                content, finish_reason, usage = await _collect_truncated_response(
                    state, request_id, output_queue, stop_sequences
                )
                return {
                    "text": content,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
            finally:
                state.close_listener(request_id)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def text_completions(payload: CompletionRequest, request: Request):
        if payload.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")

        stop_sequences = _normalize_stop_sequences(payload.stop)
        created = int(time.time())
        request_id, output_queue = state.open_request(
            payload.prompt,
            _make_sampling_params(payload, stop_sequences),
        )

        if not payload.stream:
            try:
                content, finish_reason, usage = await _collect_truncated_response(
                    state, request_id, output_queue, stop_sequences
                )
                return _build_text_completion_payload(
                    request_id,
                    config.model,
                    created,
                    content,
                    finish_reason,
                    usage,
                )
            finally:
                state.close_listener(request_id)

        async def stream():
            async for frame in _stream_with_stop_handling(
                state=state,
                request=request,
                request_id=request_id,
                output_queue=output_queue,
                stop_sequences=stop_sequences,
                render_chunk=lambda chunk, text, finish_reason: _sse_json_frame(
                    _build_text_stream_payload(
                        request_id,
                        config.model,
                        created,
                        text,
                        finish_reason,
                        usage=build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                        if _should_include_usage(
                            payload.stream_options,
                            finished=chunk.finished or finish_reason == "stop",
                        )
                        else None,
                    )
                ),
                render_tail=lambda tail: _sse_json_frame(
                    _build_text_stream_payload(request_id, config.model, created, tail, None)
                ),
                done_frame="data: [DONE]\n\n",
            ):
                yield frame

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def completions(payload: ChatCompletionRequest, request: Request):
        if payload.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")

        stop_sequences = _normalize_stop_sequences(payload.stop)
        created = int(time.time())
        # Pass messages list so the tokenizer worker can apply the chat template.
        # Fall back to flattened prompt if messages is not provided.
        chat_messages = (
            [{"role": m.role, "content": m.content} for m in payload.messages]
            if payload.messages
            else None
        )
        request_id, output_queue = state.open_request(
            _flatten_prompt(payload),
            _make_sampling_params(payload, stop_sequences),
            messages=chat_messages,
        )

        if not payload.stream:
            try:
                content, finish_reason, usage = await _collect_truncated_response(
                    state, request_id, output_queue, stop_sequences
                )
                return _build_completion_payload(
                    request_id,
                    config.model,
                    created,
                    content,
                    finish_reason,
                    usage,
                )
            finally:
                state.close_listener(request_id)

        async def stream():
            first = True

            def render_chunk(chunk: StreamChunk, text: str, finish_reason: str | None) -> str:
                nonlocal first
                delta = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if text:
                    delta["content"] = text
                return _sse_json_frame(
                    _build_stream_payload(
                        request_id,
                        config.model,
                        created,
                        delta,
                        finish_reason,
                        usage=build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                        if _should_include_usage(
                            payload.stream_options,
                            finished=chunk.finished or finish_reason == "stop",
                        )
                        else None,
                    )
                )

            async for frame in _stream_with_stop_handling(
                state=state,
                request=request,
                request_id=request_id,
                output_queue=output_queue,
                stop_sequences=stop_sequences,
                render_chunk=render_chunk,
                render_tail=lambda tail: _sse_json_frame(
                    _build_stream_payload(request_id, config.model, created, {"content": tail}, None)
                ),
                done_frame="data: [DONE]\n\n",
            ):
                yield frame

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
