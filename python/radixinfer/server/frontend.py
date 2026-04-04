from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

from radixinfer.config import ServerConfig
from radixinfer.server.common import build_usage
from radixinfer.transport.protocol import (
    AbortRequest,
    BatchStreamChunk,
    SamplingParams,
    StreamChunk,
    TokenizeRequest,
)
from radixinfer.transport.queues import make_zmq_async_pull, make_zmq_push
from radixinfer.transport.tokenizer_worker import start_tokenizer_process
from radixinfer.utils.mp import has_zmq


@dataclass
class FrontendManager:
    config: ServerConfig
    tokenizer_ingress: Any = field(default_factory=mp.Queue)
    runtime_ingress: Any = field(default_factory=mp.Queue)
    frontend_queue: Any = field(default_factory=mp.Queue)
    request_counter: int = 0
    tokenizer_process: mp.Process | None = None
    runtime_processes: list[mp.Process] = field(default_factory=list)
    listeners: dict[int, asyncio.Queue[StreamChunk]] = field(default_factory=dict)
    listen_task: asyncio.Task | None = None
    backend_started: bool = False

    def _stop_process(self, process: mp.Process | None) -> bool:
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
        self._stop_processes(self.runtime_processes)
        self._stop_process(self.tokenizer_process)
        self.runtime_processes = []
        self.tokenizer_process = None
        self.backend_started = False
        self._destroy_zmq_context()

    def _use_zmq(self) -> bool:
        return self.config.use_zmq and has_zmq()

    def _stop_processes(
        self,
        processes: list[mp.Process],
        *,
        deadline: float | None = None,
    ) -> bool:
        for process in processes:
            timeout = None
            if deadline is not None:
                timeout = max(0.1, deadline - time.monotonic())
            process.join(timeout=timeout)
        all_stopped = True
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                try:
                    process.kill()
                except Exception:
                    pass
                process.join(timeout=5)
            all_stopped = all_stopped and not process.is_alive()
        return all_stopped

    def start_backend(self) -> None:
        if self.backend_started:
            return

        ack_queue: mp.Queue | None = None
        try:
            use_zmq = self._use_zmq()
            if use_zmq:
                self.tokenizer_ingress = make_zmq_push(self.config.zmq_tokenizer_addr, create=False)
                self.runtime_ingress = make_zmq_push(self.config.zmq_backend_addr, create=False)
                self.frontend_queue = make_zmq_async_pull(self.config.zmq_frontend_addr, create=True)
            if self.tokenizer_process is None:
                self.tokenizer_process = start_tokenizer_process(
                    self.config.zmq_tokenizer_addr if use_zmq else self.tokenizer_ingress,
                    self.config.zmq_backend_addr if use_zmq else self.runtime_ingress,
                    self.config.zmq_frontend_addr if use_zmq else self.frontend_queue,
                    self.config.tokenizer_name or self.config.model,
                )
            if not self.runtime_processes:
                from radixinfer.runtime.scheduler import start_runtime_process

                ack_queue = mp.Queue()
                self.runtime_processes = start_runtime_process(
                    self.config,
                    ack_queue=ack_queue,
                )
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    dead = [process for process in self.runtime_processes if not process.is_alive()]
                    if dead:
                        codes = [process.exitcode for process in dead]
                        raise RuntimeError(
                            f"Runtime process(es) exited unexpectedly during startup "
                            f"(exit codes: {codes}). "
                            f"Check for port conflicts (--dist-port={self.config.dist_port}) or model load errors."
                        )
                    try:
                        ack_queue.get(timeout=0.5)
                        break
                    except Exception:
                        continue
                else:
                    raise RuntimeError(
                        f"Runtime process did not signal readiness within 300 seconds "
                        f"(dist-port={self.config.dist_port})."
                    )
            self.backend_started = True
        except Exception:
            self._cleanup_startup_failure()
            raise
        finally:
            if ack_queue is not None:
                ack_queue.close()
                ack_queue.join_thread()

    def start(self) -> None:
        self.start_backend()

    async def start_listener(self) -> None:
        if self.listen_task is None:
            self.listen_task = asyncio.create_task(self._listen_frontend())

    async def _listen_frontend(self) -> None:
        import inspect
        from queue import Empty

        get = self.frontend_queue.get
        is_async = inspect.iscoroutinefunction(get)
        try:
            while True:
                if is_async:
                    msg = await get()
                else:
                    try:
                        msg = self.frontend_queue.get_nowait()
                    except Empty:
                        await asyncio.sleep(0.001)
                        continue
                if msg is None:
                    return
                chunks = msg.chunks if isinstance(msg, BatchStreamChunk) else [msg]
                for chunk in chunks:
                    queue = self.listeners.get(chunk.request_id)
                    if queue is not None:
                        await queue.put(chunk)
        except asyncio.CancelledError:
            return

    def next_request_id(self) -> int:
        current = self.request_counter
        self.request_counter += 1
        return current

    def open_listener(self) -> tuple[int, asyncio.Queue[StreamChunk]]:
        request_id = self.next_request_id()
        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        self.listeners[request_id] = output_queue
        return request_id, output_queue

    def close_listener(self, request_id: int) -> None:
        self.listeners.pop(request_id, None)

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

    def open_request(
        self,
        prompt: str,
        sampling: SamplingParams,
        *,
        messages: list | None = None,
    ) -> tuple[int, asyncio.Queue[StreamChunk]]:
        request_id, output_queue = self.open_listener()
        self.submit_request(
            request_id,
            prompt,
            sampling,
            messages=messages,
        )
        return request_id, output_queue

    async def abort_request(self, request_id: int) -> None:
        self.runtime_ingress.put(AbortRequest(request_id=request_id))

    async def collect_response(
        self,
        request_id: int,
        output_queue: asyncio.Queue[StreamChunk],
    ) -> tuple[str, str, dict[str, int]]:
        parts: list[str] = []
        finish_reason = "stop"
        usage = build_usage(0, 0)
        while True:
            chunk = await output_queue.get()
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finished:
                finish_reason = chunk.finish_reason
                usage = build_usage(chunk.prompt_tokens or 0, chunk.completion_tokens or 0)
                break
        return "".join(parts), finish_reason, usage

    async def shutdown(self) -> None:
        if self.backend_started:
            self.tokenizer_ingress.put(None)
            self.runtime_ingress.put(None)
        if self.listen_task is not None:
            self.listen_task.cancel()
            await asyncio.gather(self.listen_task, return_exceptions=True)
        tokenizer_stopped = self._stop_process(self.tokenizer_process)
        deadline = time.monotonic() + 30
        runtime_stopped = self._stop_processes(self.runtime_processes, deadline=deadline)
        self.tokenizer_process = None
        self.runtime_processes = []
        self.listen_task = None
        self.backend_started = False
        self._destroy_zmq_context()
        if not tokenizer_stopped or not runtime_stopped:
            print(
                "Warning: forced shutdown was required for one or more worker processes."
            )
