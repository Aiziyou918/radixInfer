from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from fastapi.exceptions import HTTPException
except ModuleNotFoundError:
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from radixinfer.config import ServerConfig
from radixinfer.server.common import build_usage
from radixinfer.transport.protocol import StreamChunk
from radixinfer.server.frontend import FrontendManager
from radixinfer.api.rendering import (
    build_completion_payload as _build_completion_payload,
    build_stream_payload as _build_stream_payload,
    build_text_completion_payload as _build_text_completion_payload,
    build_text_stream_payload as _build_text_stream_payload,
    collect_truncated_response as _collect_truncated_response,
    flatten_prompt as _flatten_prompt,
    make_sampling_params as _make_sampling_params,
    normalize_stop_sequences as _normalize_stop_sequences,
    should_include_usage as _should_include_usage,
    sse_json_frame as _sse_json_frame,
    sse_text_frame as _sse_text_frame,
    stream_with_stop_handling as _stream_with_stop_handling,
)
from radixinfer.api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    GenerateRequest,
)


AppState = FrontendManager


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
