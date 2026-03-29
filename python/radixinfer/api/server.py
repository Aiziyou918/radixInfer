from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

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
from radixinfer.runtime.scheduler import start_runtime_process
from radixinfer.transport.queues import make_zmq_pull, make_zmq_push
from radixinfer.transport.protocol import (
    AbortRequest,
    SamplingParams,
    StreamChunk,
    TokenizeRequest,
)
from radixinfer.transport.tokenizer_worker import start_tokenizer_process
from radixinfer.utils.mp import has_zmq


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


@dataclass
class AppState:
    config: ServerConfig
    tokenizer_ingress: Any = field(default_factory=mp.Queue)
    runtime_ingress: Any = field(default_factory=mp.Queue)
    frontend_queue: Any = field(default_factory=mp.Queue)
    request_counter: int = 0
    tokenizer_process: mp.Process | None = None
    runtime_processes: list = field(default_factory=list)
    listeners: dict[int, asyncio.Queue[StreamChunk]] = field(default_factory=dict)
    listen_task: asyncio.Task | None = None

    def _stop_process(self, process: mp.Process | None, *, name: str) -> bool:
        if process is None:
            return True
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=5)
        return not process.is_alive()

    def _destroy_zmq_context(self) -> None:
        try:
            import zmq

            zmq.Context.instance().destroy(linger=0)
        except Exception:
            pass

    def _cleanup_startup_failure(self) -> None:
        for p in self.runtime_processes:
            self._stop_process(p, name="runtime")
        self._stop_process(self.tokenizer_process, name="tokenizer")
        self.runtime_processes = []
        self.tokenizer_process = None
        self._destroy_zmq_context()

    def start(self) -> None:
        ack_queue: mp.Queue | None = None
        try:
            if self.config.use_zmq and has_zmq():
                self.tokenizer_ingress = make_zmq_push(self.config.zmq_tokenizer_addr, create=False)
                self.runtime_ingress = make_zmq_push(self.config.zmq_backend_addr, create=False)
                self.frontend_queue = make_zmq_pull(self.config.zmq_frontend_addr, create=True)
            if self.tokenizer_process is None:
                self.tokenizer_process = start_tokenizer_process(
                    self.config.zmq_tokenizer_addr if self.config.use_zmq and has_zmq() else self.tokenizer_ingress,
                    self.config.zmq_backend_addr if self.config.use_zmq and has_zmq() else self.runtime_ingress,
                    self.config.zmq_frontend_addr if self.config.use_zmq and has_zmq() else self.frontend_queue,
                    self.config.tokenizer_name or self.config.model,
                )
            if not self.runtime_processes:
                import time

                ack_queue = mp.Queue()
                self.runtime_processes = start_runtime_process(
                    self.config,
                    self.config.zmq_backend_addr if self.config.use_zmq and has_zmq() else self.runtime_ingress,
                    self.config.zmq_tokenizer_addr if self.config.use_zmq and has_zmq() else self.tokenizer_ingress,
                    ack_queue=ack_queue,
                )
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    dead = [p for p in self.runtime_processes if not p.is_alive()]
                    if dead:
                        codes = [p.exitcode for p in dead]
                        raise RuntimeError(
                            f"Runtime process(es) exited unexpectedly during startup "
                            f"(exit codes: {codes}). "
                            f"Check for port conflicts (--dist-port={self.config.dist_port}) or model load errors."
                        )
                    try:
                        ack_queue.get(timeout=0.5)
                        return
                    except Exception:
                        continue
                raise RuntimeError(
                    f"Runtime process did not signal readiness within 300 seconds "
                    f"(dist-port={self.config.dist_port})."
                )
        except Exception:
            self._cleanup_startup_failure()
            raise
        finally:
            if ack_queue is not None:
                ack_queue.close()
                ack_queue.join_thread()

    async def start_listener(self) -> None:
        if self.listen_task is None:
            self.listen_task = asyncio.create_task(self._listen_frontend())

    async def _listen_frontend(self) -> None:
        from queue import Empty

        try:
            while True:
                try:
                    chunk = self.frontend_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.001)
                    continue
                if chunk is None:
                    return
                queue = self.listeners.get(chunk.request_id)
                if queue is not None:
                    await queue.put(chunk)
        except asyncio.CancelledError:
            return

    def next_request_id(self) -> int:
        current = self.request_counter
        self.request_counter += 1
        return current

    def submit_request(
        self,
        request_id: int,
        prompt: str,
        sampling: SamplingParams,
        messages: list | None = None,
    ) -> None:
        self.tokenizer_ingress.put(
            TokenizeRequest(
                request_id=request_id,
                prompt=prompt,
                sampling=sampling,
                messages=messages,
            )
        )

    async def abort_request(self, request_id: int) -> None:
        self.runtime_ingress.put(AbortRequest(request_id=request_id))

    async def collect_response(
        self,
        request_id: int,
        output_queue: asyncio.Queue[StreamChunk],
    ) -> tuple[str, str, dict[str, int]]:
        parts: list[str] = []
        finish_reason = "stop"
        usage = _build_usage(0, 0)
        while True:
            chunk = await output_queue.get()
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finished:
                finish_reason = chunk.finish_reason
                usage = _build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                break
        return "".join(parts), finish_reason, usage

    async def shutdown(self) -> None:
        self.tokenizer_ingress.put(None)
        self.runtime_ingress.put(None)
        if self.listen_task is not None:
            self.listen_task.cancel()
            await asyncio.gather(self.listen_task, return_exceptions=True)
        tokenizer_stopped = self._stop_process(self.tokenizer_process, name="tokenizer")
        # Wait for all runtime processes together so TP ranks can reach sync_all_ranks()
        # simultaneously and coordinate destroy_process_group() cleanly.
        deadline = time.monotonic() + 30
        for p in self.runtime_processes:
            remaining = max(0.1, deadline - time.monotonic())
            p.join(timeout=remaining)
        for i, p in enumerate(self.runtime_processes):
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass
                p.join(timeout=5)
        runtime_stopped = all(not p.is_alive() for p in self.runtime_processes)
        self.tokenizer_process = None
        self.runtime_processes = []
        self._destroy_zmq_context()
        if not tokenizer_stopped or not runtime_stopped:
            print(
                "Warning: forced shutdown was required for one or more worker processes."
            )


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


def _build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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


def create_app(config: ServerConfig, state: AppState | None = None) -> FastAPI:
    if FastAPI is None or Request is None or StreamingResponse is None:
        raise RuntimeError("FastAPI dependencies are not installed. Install project dependencies first.")
    if state is None:
        state = AppState(config=config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
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
        request_id = state.next_request_id()
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        state.listeners[request_id] = output_queue
        state.submit_request(
            request_id,
            payload.prompt,
            SamplingParams(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                ignore_eos=payload.ignore_eos,
                stop=stop_sequences,
            ),
        )

        async def stream():
            matcher = _StreamingStopState(stop_sequences)
            try:
                while True:
                    if await request.is_disconnected():
                        await state.abort_request(request_id)
                        break
                    chunk = await output_queue.get()
                    text = chunk.text
                    matched = False
                    if text:
                        text, matched = matcher.push(text)
                    if text:
                        yield f"data: {text}\n"
                    if matched:
                        await state.abort_request(request_id)
                        yield "data: [DONE]\n"
                        break
                    if chunk.finished:
                        tail = matcher.flush()
                        if tail:
                            yield f"data: {tail}\n"
                        yield "data: [DONE]\n"
                        break
            finally:
                state.listeners.pop(request_id, None)

        if not payload.stream:
            try:
                content, finish_reason, usage = await state.collect_response(request_id, output_queue)
                content, matched_stop = _truncate_for_stop_sequences(content, stop_sequences)
                if matched_stop:
                    finish_reason = "stop"
                return {
                    "text": content,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
            finally:
                state.listeners.pop(request_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def text_completions(payload: CompletionRequest, request: Request):
        if payload.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")

        stop_sequences = _normalize_stop_sequences(payload.stop)
        request_id = state.next_request_id()
        created = int(time.time())
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        state.listeners[request_id] = output_queue
        state.submit_request(
            request_id,
            payload.prompt,
            SamplingParams(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                ignore_eos=payload.ignore_eos,
                stop=stop_sequences,
                n=payload.n,
                presence_penalty=payload.presence_penalty,
                frequency_penalty=payload.frequency_penalty,
            ),
        )

        if not payload.stream:
            try:
                content, finish_reason, usage = await state.collect_response(request_id, output_queue)
                content, matched_stop = _truncate_for_stop_sequences(content, stop_sequences)
                if matched_stop:
                    finish_reason = "stop"
                return _build_text_completion_payload(
                    request_id,
                    config.model,
                    created,
                    content,
                    finish_reason,
                    usage,
                )
            finally:
                state.listeners.pop(request_id, None)

        async def stream():
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
                    body = _build_text_stream_payload(
                        request_id,
                        config.model,
                        created,
                        text,
                        finish_reason,
                        usage=(
                            _build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                            if (chunk.finished or matched_stop)
                            and payload.stream_options
                            and payload.stream_options.include_usage
                            else None
                        ),
                    )
                    yield f"data: {json.dumps(body)}\n\n"
                    if matched_stop:
                        await state.abort_request(request_id)
                        break
                    if chunk.finished:
                        tail = matcher.flush()
                        if tail:
                            yield f"data: {json.dumps(_build_text_stream_payload(request_id, config.model, created, tail, None))}\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                state.listeners.pop(request_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def completions(payload: ChatCompletionRequest, request: Request):
        if payload.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")

        stop_sequences = _normalize_stop_sequences(payload.stop)
        request_id = state.next_request_id()
        created = int(time.time())
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        state.listeners[request_id] = output_queue
        # Pass messages list so the tokenizer worker can apply the chat template.
        # Fall back to flattened prompt if messages is not provided.
        chat_messages = (
            [{"role": m.role, "content": m.content} for m in payload.messages]
            if payload.messages
            else None
        )
        state.submit_request(
            request_id,
            _flatten_prompt(payload),
            SamplingParams(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                ignore_eos=payload.ignore_eos,
                stop=stop_sequences,
                n=payload.n,
                presence_penalty=payload.presence_penalty,
                frequency_penalty=payload.frequency_penalty,
            ),
            messages=chat_messages,
        )

        if not payload.stream:
            try:
                content, finish_reason, usage = await state.collect_response(request_id, output_queue)
                content, matched_stop = _truncate_for_stop_sequences(content, stop_sequences)
                if matched_stop:
                    finish_reason = "stop"
                return _build_completion_payload(
                    request_id,
                    config.model,
                    created,
                    content,
                    finish_reason,
                    usage,
                )
            finally:
                state.listeners.pop(request_id, None)

        async def stream():
            first = True
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
                    delta = {}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    if text:
                        delta["content"] = text
                    finish_reason = None if not chunk.finished else chunk.finish_reason
                    if matched_stop:
                        finish_reason = "stop"
                    body = _build_stream_payload(
                        request_id,
                        config.model,
                        created,
                        delta,
                        finish_reason,
                        usage=(
                            _build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                            if (chunk.finished or matched_stop)
                            and payload.stream_options
                            and payload.stream_options.include_usage
                            else None
                        ),
                    )
                    yield f"data: {json.dumps(body)}\n\n"
                    if matched_stop:
                        await state.abort_request(request_id)
                        break
                    if chunk.finished:
                        tail = matcher.flush()
                        if tail:
                            yield f"data: {json.dumps(_build_stream_payload(request_id, config.model, created, {'content': tail}, None))}\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                state.listeners.pop(request_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
