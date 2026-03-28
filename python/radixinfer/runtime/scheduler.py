from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty

from radixinfer.cache.page_pool import PagePool
from radixinfer.cache.prefix_store import PrefixStore
from radixinfer.config import ServerConfig
from radixinfer.engine.base import DecodeInput
from radixinfer.engine import build_engine
from radixinfer.runtime.planner import BatchPlanner
from radixinfer.runtime.types import RequestPhase, RuntimeRequest
from radixinfer.transport.protocol import AbortRequest, DetokenizeRequest, TokenizedRequest


class SchedulerRuntime:
    def __init__(self, config: ServerConfig, ingress: mp.Queue, tokenizer_queue: mp.Queue) -> None:
        self.config = config
        self.ingress = ingress
        self.tokenizer_queue = tokenizer_queue
        self.requests: dict[int, RuntimeRequest] = {}
        self.page_pool = PagePool(total_pages=config.total_pages, page_size=config.page_size)
        self.prefix_store = PrefixStore(
            capacity=config.prefix_cache_capacity,
            page_size=config.page_size,
        )
        self.planner = BatchPlanner(
            max_batch_size=config.max_batch_size,
            max_prefill_tokens=config.max_prefill_tokens,
        )
        self.engine = build_engine(config)

    def run(self) -> None:
        while True:
            if self._drain_ingress():
                return
            self._tick()
            time.sleep(self.config.scheduler_tick_interval)

    def _drain_ingress(self) -> bool:
        stop = False
        for _ in range(self.config.max_queue_drain):
            try:
                item = self.ingress.get_nowait()
            except Empty:
                break
            if item is None:
                stop = True
                continue
            if isinstance(item, TokenizedRequest):
                request = RuntimeRequest(
                    request_id=item.request_id,
                    prompt_tokens=item.token_ids,
                    sampling=item.sampling,
                    eos_token_id=item.eos_token_id,
                    stop_token_ids=item.stop_token_ids,
                )
                request.prefix_matched = self.prefix_store.match(item.token_ids).matched_tokens
                self.requests[item.request_id] = request
            elif isinstance(item, AbortRequest):
                request = self.requests.get(item.request_id)
                if request is not None and not request.finished:
                    request.phase = RequestPhase.ABORTED
                    if request.reservation is not None:
                        self.page_pool.release(request.reservation)
                        request.reservation = None
                    self.tokenizer_queue.put(
                        DetokenizeRequest(
                            request_id=item.request_id,
                            token_id=0,
                            finished=True,
                            finish_reason="abort",
                        )
                    )
        return stop

    def _tick(self) -> None:
        plan = self.planner.build_plan(list(self.requests.values()))
        if not plan.prefill and not plan.decode:
            self._age_waiting()
            return
        if plan.prefill:
            self._run_prefill(plan.prefill)
        if plan.decode:
            self._run_decode(plan.decode)
        self._age_waiting(reset_selected=set(plan.prefill + plan.decode))

    def _run_prefill(self, request_ids: list[int]) -> None:
        remaining_budget = self.config.max_prefill_tokens
        for request_id in request_ids:
            request = self.requests[request_id]
            if request.finished or request.prefill_complete:
                continue
            if request.reservation is None:
                total_reserved = request.uncached_prompt_tokens + request.sampling.max_tokens
                reservation = self.page_pool.reserve_for_tokens(total_reserved)
                if reservation is None:
                    continue
                request.reservation = reservation
                request.reserved_tokens = total_reserved

            chunk_tokens = min(request.remaining_prefill_tokens, remaining_budget)
            if chunk_tokens <= 0:
                break

            request.phase = RequestPhase.PREFILLING
            request.prefill_cursor += chunk_tokens
            remaining_budget -= chunk_tokens

            if request.prefill_complete:
                request.phase = RequestPhase.READY_TO_DECODE
                self.prefix_store.insert(request.prompt_tokens)
            if remaining_budget <= 0:
                break

    def _run_decode(self, request_ids: list[int]) -> None:
        selected = [
            self.requests[request_id]
            for request_id in request_ids
            if self.requests[request_id].phase in {RequestPhase.READY_TO_DECODE, RequestPhase.DECODING}
        ]
        if not selected:
            return
        for request in selected:
            if request.phase == RequestPhase.READY_TO_DECODE:
                request.phase = RequestPhase.DECODING
        output = self.engine.decode(
            DecodeInput(
                request_ids=[req.request_id for req in selected],
                token_ids=[req.all_tokens for req in selected],
            )
        )
        for request, token_id in zip(selected, output.next_token_ids, strict=True):
            request.generated_tokens.append(token_id)
            finished = False
            finish_reason = "running"
            request_stop_tokens = set(self.config.stop_token_ids).union(request.stop_token_ids)
            if (
                not request.sampling.ignore_eos
                and request.eos_token_id is not None
                and token_id == request.eos_token_id
            ):
                finished = True
                finish_reason = "stop"
            elif token_id in request_stop_tokens:
                finished = True
                finish_reason = "stop"
            elif request.remaining_tokens <= 0:
                finished = True
                finish_reason = "length"
            if finished:
                request.phase = RequestPhase.FINISHED
                if request.reservation is not None:
                    self.page_pool.release(request.reservation)
                    request.reservation = None
            self.tokenizer_queue.put(
                DetokenizeRequest(
                    request_id=request.request_id,
                    token_id=token_id,
                    finished=finished,
                    finish_reason=finish_reason,
                )
            )

    def _age_waiting(self, reset_selected: set[int] | None = None) -> None:
        reset_selected = reset_selected or set()
        for request in self.requests.values():
            if request.request_id in reset_selected:
                request.age = 0
            elif not request.finished:
                request.age += 1


def start_runtime_process(config: ServerConfig, ingress: mp.Queue, tokenizer_queue: mp.Queue) -> mp.Process:
    process = mp.Process(
        target=SchedulerRuntime(config, ingress, tokenizer_queue).run,
        name="radixinfer-runtime",
        daemon=True,
    )
    process.start()
    return process
