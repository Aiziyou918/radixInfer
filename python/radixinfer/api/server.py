from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ModuleNotFoundError:
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]

    class BaseModel:  # type: ignore[no-redef]
        pass

from radixinfer.config import ServerConfig
from radixinfer.runtime.scheduler import SchedulerRuntime, start_runtime_process
from radixinfer.transport.protocol import (
    AbortRequest,
    SamplingParams,
    StreamChunk,
    TokenizeRequest,
    TokenizedRequest,
)
from radixinfer.transport.tokenizer_worker import create_tokenizer_backend, start_tokenizer_process


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
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
    runtime_process: mp.Process | None = None
    listeners: dict[int, asyncio.Queue[StreamChunk]] = field(default_factory=dict)
    listen_task: asyncio.Task | None = None
    inline_runtime: SchedulerRuntime | None = None
    inline_tokenizer: Any | None = None
    inline_tasks: set[asyncio.Task] = field(default_factory=set)

    @property
    def inline_mode(self) -> bool:
        return self.config.start_method == "inline"

    def start(self) -> None:
        if self.inline_mode:
            self.inline_tokenizer = create_tokenizer_backend(self.config.tokenizer_name or self.config.model)
            self.inline_runtime = SchedulerRuntime(
                self.config,
                self.runtime_ingress,
                self.tokenizer_ingress,
            )
            return
        if self.tokenizer_process is None:
            self.tokenizer_process = start_tokenizer_process(
                self.tokenizer_ingress,
                self.runtime_ingress,
                self.frontend_queue,
                self.config.tokenizer_name or self.config.model,
            )
        if self.runtime_process is None:
            self.runtime_process = start_runtime_process(
                self.config,
                self.runtime_ingress,
                self.tokenizer_ingress,
            )

    async def start_listener(self) -> None:
        if self.inline_mode:
            return
        if self.listen_task is None:
            self.listen_task = asyncio.create_task(self._listen_frontend())

    async def _listen_frontend(self) -> None:
        while True:
            chunk = await asyncio.to_thread(self.frontend_queue.get)
            if chunk is None:
                return
            queue = self.listeners.get(chunk.request_id)
            if queue is not None:
                await queue.put(chunk)

    def next_request_id(self) -> int:
        current = self.request_counter
        self.request_counter += 1
        return current

    def submit_request(self, request_id: int, prompt: str, sampling: SamplingParams) -> None:
        if self.inline_mode:
            task = asyncio.create_task(self._run_inline_request(request_id, prompt, sampling))
            self.inline_tasks.add(task)
            task.add_done_callback(self.inline_tasks.discard)
            return
        self.tokenizer_ingress.put(
            TokenizeRequest(
                request_id=request_id,
                prompt=prompt,
                sampling=sampling,
            )
        )

    async def abort_request(self, request_id: int) -> None:
        self.runtime_ingress.put(AbortRequest(request_id=request_id))
        if not self.inline_mode:
            return
        await self._pump_inline_runtime()

    async def _run_inline_request(
        self,
        request_id: int,
        prompt: str,
        sampling: SamplingParams,
    ) -> None:
        assert self.inline_runtime is not None
        assert self.inline_tokenizer is not None
        self.runtime_ingress.put(
            TokenizedRequest(
                request_id=request_id,
                token_ids=self.inline_tokenizer.encode(prompt),
                sampling=sampling,
                eos_token_id=getattr(self.inline_tokenizer, "eos_token_id", None),
                stop_token_ids=getattr(self.inline_tokenizer, "stop_token_ids", ()),
            )
        )
        while True:
            done = await self._pump_inline_runtime()
            if done.get(request_id, False):
                return
            await asyncio.sleep(self.config.scheduler_tick_interval)

    async def _pump_inline_runtime(self) -> dict[int, bool]:
        assert self.inline_runtime is not None
        assert self.inline_tokenizer is not None
        completed: dict[int, bool] = {}
        self.inline_runtime._drain_ingress()
        self.inline_runtime._tick()
        while True:
            try:
                detok = self.tokenizer_ingress.get_nowait()
            except Empty:
                break
            if detok is None:
                continue
            chunk = StreamChunk(
                request_id=detok.request_id,
                token_id=detok.token_id,
                text=self.inline_tokenizer.decode_token(detok.token_id) if detok.emit_text else "",
                finished=detok.finished,
                finish_reason=detok.finish_reason,
                prompt_tokens=detok.prompt_tokens,
                completion_tokens=detok.completion_tokens,
            )
            listener = self.listeners.get(chunk.request_id)
            if listener is not None:
                await listener.put(chunk)
            if chunk.finished:
                completed[chunk.request_id] = True
        return completed

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
        if self.inline_tasks:
            await asyncio.gather(*list(self.inline_tasks), return_exceptions=True)
        if self.inline_mode:
            return
        self.tokenizer_ingress.put(None)
        self.runtime_ingress.put(None)
        self.frontend_queue.put(None)
        if self.listen_task is not None:
            await self.listen_task
        if self.tokenizer_process is not None:
            self.tokenizer_process.join(timeout=1)
        if self.runtime_process is not None:
            self.runtime_process.join(timeout=1)


def _flatten_prompt(request: ChatCompletionRequest) -> str:
    if request.messages:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)
    return request.prompt or ""


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


def create_app(config: ServerConfig) -> FastAPI:
    if FastAPI is None or Request is None or StreamingResponse is None:
        raise RuntimeError("FastAPI dependencies are not installed. Install project dependencies first.")
    if config.start_method == "inline":
        state = AppState(
            config=config,
            tokenizer_ingress=Queue(),
            runtime_ingress=Queue(),
            frontend_queue=Queue(),
        )
    else:
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
        return {"object": "list", "data": [{"id": config.model, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def completions(payload: ChatCompletionRequest, request: Request):
        request_id = state.next_request_id()
        created = int(time.time())
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        state.listeners[request_id] = output_queue
        state.submit_request(
            request_id,
            _flatten_prompt(payload),
            SamplingParams(
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_k=payload.top_k,
                top_p=payload.top_p,
                ignore_eos=payload.ignore_eos,
            ),
        )

        if not payload.stream:
            try:
                content, finish_reason, usage = await state.collect_response(request_id, output_queue)
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
            try:
                while True:
                    if await request.is_disconnected():
                        await state.abort_request(request_id)
                        break
                    chunk = await output_queue.get()
                    delta = {}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    if chunk.text:
                        delta["content"] = chunk.text
                    body = _build_stream_payload(
                        request_id,
                        config.model,
                        created,
                        delta,
                        None if not chunk.finished else chunk.finish_reason,
                        usage=(
                            _build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                            if chunk.finished and payload.stream_options and payload.stream_options.include_usage
                            else None
                        ),
                    )
                    yield f"data: {json.dumps(body)}\n\n"
                    if chunk.finished:
                        break
                yield "data: [DONE]\n\n"
            finally:
                state.listeners.pop(request_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
