from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

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
from radixinfer.runtime.scheduler import start_runtime_process
from radixinfer.transport.protocol import AbortRequest, SamplingParams, StreamChunk, TokenizeRequest
from radixinfer.transport.tokenizer_worker import start_tokenizer_process


class ChatMessage(BaseModel):
    role: str
    content: str


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


@dataclass
class AppState:
    config: ServerConfig
    tokenizer_ingress: mp.Queue = field(default_factory=mp.Queue)
    runtime_ingress: mp.Queue = field(default_factory=mp.Queue)
    frontend_queue: mp.Queue = field(default_factory=mp.Queue)
    request_counter: int = 0
    tokenizer_process: mp.Process | None = None
    runtime_process: mp.Process | None = None
    listeners: dict[int, asyncio.Queue[StreamChunk]] = field(default_factory=dict)
    listen_task: asyncio.Task | None = None

    def start(self) -> None:
        if self.tokenizer_process is None:
            self.tokenizer_process = start_tokenizer_process(
                self.tokenizer_ingress,
                self.runtime_ingress,
                self.frontend_queue,
            )
        if self.runtime_process is None:
            self.runtime_process = start_runtime_process(
                self.config,
                self.runtime_ingress,
                self.tokenizer_ingress,
            )

    async def start_listener(self) -> None:
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

    async def shutdown(self) -> None:
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


def create_app(config: ServerConfig) -> FastAPI:
    if FastAPI is None or Request is None or StreamingResponse is None:
        raise RuntimeError("FastAPI dependencies are not installed. Install project dependencies first.")
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
    async def completions(payload: ChatCompletionRequest, request: Request) -> StreamingResponse:
        request_id = state.next_request_id()
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        state.listeners[request_id] = output_queue
        state.tokenizer_ingress.put(
            TokenizeRequest(
                request_id=request_id,
                prompt=_flatten_prompt(payload),
                sampling=SamplingParams(
                    max_tokens=payload.max_tokens,
                    temperature=payload.temperature,
                    top_k=payload.top_k,
                    top_p=payload.top_p,
                    ignore_eos=payload.ignore_eos,
                ),
            )
        )

        async def stream():
            first = True
            try:
                while True:
                    if await request.is_disconnected():
                        state.runtime_ingress.put(AbortRequest(request_id=request_id))
                        break
                    chunk = await output_queue.get()
                    delta = {}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    if chunk.text:
                        delta["content"] = chunk.text
                    body = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None if not chunk.finished else chunk.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(body)}\n\n"
                    if chunk.finished:
                        break
                yield "data: [DONE]\n\n"
            finally:
                state.listeners.pop(request_id, None)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
