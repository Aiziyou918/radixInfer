from __future__ import annotations

import asyncio
import json
from typing import Callable

from radixinfer.server.frontend import FrontendManager
from radixinfer.transport.protocol import SamplingParams, StreamChunk

from .schemas import ChatCompletionRequest, StreamOptions


def flatten_prompt(request: ChatCompletionRequest) -> str:
    if request.messages:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
    return request.prompt or ""


def normalize_stop_sequences(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,) if stop else ()
    return tuple(item for item in stop if item)


def truncate_for_stop_sequences(text: str, stop_sequences: tuple[str, ...]) -> tuple[str, bool]:
    stop_idx: int | None = None
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx >= 0 and (stop_idx is None or idx < stop_idx):
            stop_idx = idx
    if stop_idx is None:
        return text, False
    return text[:stop_idx], True


class StreamingStopState:
    def __init__(self, stop_sequences: tuple[str, ...]) -> None:
        self.stop_sequences = stop_sequences
        self._pending = ""
        self._tail_len = max((len(stop) for stop in stop_sequences), default=0) - 1

    def push(self, text: str) -> tuple[str, bool]:
        if not self.stop_sequences:
            return text, False
        combined = self._pending + text
        truncated, matched = truncate_for_stop_sequences(combined, self.stop_sequences)
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


def build_stream_payload(
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


def build_completion_payload(
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


def build_text_completion_payload(
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


def build_text_stream_payload(
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


def make_sampling_params(payload, stop_sequences: tuple[str, ...]) -> SamplingParams:
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


def should_include_usage(stream_options: StreamOptions | None, *, finished: bool) -> bool:
    return finished and stream_options is not None and stream_options.include_usage


def sse_text_frame(text: str) -> str:
    return f"data: {text}\n"


def sse_json_frame(body: dict) -> str:
    return f"data: {json.dumps(body)}\n\n"


async def stream_with_stop_handling(
    *,
    state: FrontendManager,
    request,
    request_id: int,
    output_queue: asyncio.Queue[StreamChunk],
    stop_sequences: tuple[str, ...],
    render_chunk: Callable[[StreamChunk, str, str | None], str | None],
    render_tail: Callable[[str], str | None],
    done_frame: str,
):
    matcher = StreamingStopState(stop_sequences)
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


async def collect_truncated_response(
    state: FrontendManager,
    request_id: int,
    output_queue: asyncio.Queue[StreamChunk],
    stop_sequences: tuple[str, ...],
) -> tuple[str, str, dict[str, int]]:
    content, finish_reason, usage = await state.collect_response(request_id, output_queue)
    content, matched_stop = truncate_for_stop_sequences(content, stop_sequences)
    if matched_stop:
        finish_reason = "stop"
    return content, finish_reason, usage


__all__ = [
    "StreamingStopState",
    "build_completion_payload",
    "build_stream_payload",
    "build_text_completion_payload",
    "build_text_stream_payload",
    "collect_truncated_response",
    "flatten_prompt",
    "make_sampling_params",
    "normalize_stop_sequences",
    "should_include_usage",
    "sse_json_frame",
    "sse_text_frame",
    "stream_with_stop_handling",
    "truncate_for_stop_sequences",
]
