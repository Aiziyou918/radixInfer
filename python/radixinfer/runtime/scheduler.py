from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty

from radixinfer.cache.page_pool import PagePool
from radixinfer.cache.prefix_store import PrefixStore
from radixinfer.config import ServerConfig
from radixinfer.engine.base import DecodeInput, PrefillInput
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
        self.page_pool = PagePool(
            total_pages=config.total_pages,
            page_size=config.page_size,
            kv_cache_dim=config.kv_cache_dim,
            kv_num_layers=config.kv_num_layers,
            kv_num_heads=config.kv_num_heads,
        )
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
                prefix_hit = self.prefix_store.match(item.token_ids)
                request.prefix_matched = prefix_hit.matched_tokens
                request.prefix_span = prefix_hit.cached_span
                request.prefix_cache_key = prefix_hit.cache_key
                self.prefix_store.lock(request.prefix_cache_key)
                self.requests[item.request_id] = request
            elif isinstance(item, AbortRequest):
                request = self.requests.get(item.request_id)
                if request is not None and not request.finished:
                    request.phase = RequestPhase.ABORTED
                    self.prefix_store.unlock(request.prefix_cache_key)
                    if request.reservation is not None:
                        self.page_pool.release(request.reservation)
                        request.reservation = None
                    self.tokenizer_queue.put(
                        DetokenizeRequest(
                            request_id=item.request_id,
                            token_id=0,
                            finished=True,
                            finish_reason="abort",
                            emit_text=False,
                            prompt_tokens=len(request.prompt_tokens),
                            completion_tokens=len(request.generated_tokens),
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
            if request.finished:
                continue
            if request.reservation is None:
                total_reserved = len(request.prompt_tokens) + request.sampling.max_tokens
                reservation = self._reserve_with_cache_evict(
                    total_reserved,
                    prefix_span=request.prefix_span,
                )
                if reservation is None:
                    continue
                request.reservation = reservation
                request.reserved_tokens = total_reserved
                if request.prefix_span is not None:
                    request.cache_span = request.prefix_span
            if request.prefill_complete:
                request.phase = RequestPhase.READY_TO_DECODE
                if request.prefix_span is not None:
                    request.cache_span = request.prefix_span
                continue

            chunk_tokens = min(request.remaining_prefill_tokens, remaining_budget)
            if chunk_tokens <= 0:
                break

            request.phase = RequestPhase.PREFILLING
            request.prefill_cursor += chunk_tokens
            remaining_budget -= chunk_tokens

            if request.prefill_complete:
                request.phase = RequestPhase.READY_TO_DECODE
                prefix_tokens = request.prompt_tokens[: request.prefix_matched + request.prefill_cursor]
                cache_span = self.page_pool.write_tokens(
                    request.reservation,
                    prefix_tokens[request.prefix_matched :],
                    start_offset=request.prefix_matched,
                )
                kv_prefix = (
                    self.page_pool.read_kv(request.cache_span, token_count=request.prefix_matched)
                    if request.cache_span is not None and request.prefix_matched > 0
                    else None
                )
                prefill_output = self.engine.prefill(
                    PrefillInput(
                        request_ids=[request.request_id],
                        token_ids=[prefix_tokens[request.prefix_matched :]],
                        kv_caches=[kv_prefix] if kv_prefix is not None else [],
                    )
                )
                if prefill_output.kv_writes:
                    kv_write = prefill_output.kv_writes[0]
                    cache_span = self.page_pool.write_kv(
                        request.reservation,
                        kv_write.keys,
                        kv_write.values,
                        start_offset=request.prefix_matched,
                    )
                request.cache_span = cache_span
                request.prefix_span = cache_span
                self.page_pool.share_span(cache_span)
                new_key, evicted_spans = self.prefix_store.insert(prefix_tokens, cache_span)
                for evicted_span in evicted_spans:
                    self.page_pool.evict_shared(evicted_span)
                if new_key is not None and new_key != request.prefix_cache_key:
                    self.prefix_store.unlock(request.prefix_cache_key)
                    request.prefix_cache_key = new_key
                    self.prefix_store.lock(request.prefix_cache_key)
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
            if request.cache_span is None:
                raise RuntimeError("decode request is missing active cache state")
        output = self.engine.decode(
            DecodeInput(
                request_ids=[req.request_id for req in selected],
                token_ids=[self.page_pool.read_span(req.cache_span)[-1:] for req in selected],  # type: ignore[arg-type]
                kv_caches=[
                    self.page_pool.read_kv(req.cache_span, token_count=max(0, req.cached_token_count - 1))
                    for req in selected
                ],  # type: ignore[arg-type]
            )
        )
        kv_writes = getattr(output, "kv_writes", [None] * len(selected))
        for request, token_id, kv_write in zip(
            selected, output.next_token_ids, kv_writes, strict=True
        ):
            request.generated_tokens.append(token_id)
            if request.reservation is None or request.cache_span is None:
                raise RuntimeError("decode request is missing active cache state")
            if kv_write is not None and kv_write.token_count > 0:
                request.cache_span = self.page_pool.write_kv(
                    request.reservation,
                    kv_write.keys,
                    kv_write.values,
                    start_offset=max(0, request.cached_token_count - kv_write.token_count),
                )
            request.cache_span = self.page_pool.write_tokens(
                request.reservation,
                [token_id],
                start_offset=request.cached_token_count,
            )
            finished = False
            finish_reason = "running"
            emit_text = True
            request_stop_tokens = set(self.config.stop_token_ids).union(request.stop_token_ids)
            if (
                not request.sampling.ignore_eos
                and request.eos_token_id is not None
                and token_id == request.eos_token_id
            ):
                finished = True
                finish_reason = "stop"
                emit_text = False
            elif token_id in request_stop_tokens:
                finished = True
                finish_reason = "stop"
                emit_text = False
            elif request.remaining_tokens <= 0:
                finished = True
                finish_reason = "length"
            if finished:
                request.phase = RequestPhase.FINISHED
                self.prefix_store.unlock(request.prefix_cache_key)
                if request.reservation is not None:
                    self.page_pool.release(request.reservation)
                    request.reservation = None
            self.tokenizer_queue.put(
                DetokenizeRequest(
                    request_id=request.request_id,
                    token_id=token_id,
                    finished=finished,
                    finish_reason=finish_reason,
                    emit_text=emit_text,
                    prompt_tokens=len(request.prompt_tokens),
                    completion_tokens=len(request.generated_tokens),
                )
            )

    def _age_waiting(self, reset_selected: set[int] | None = None) -> None:
        reset_selected = reset_selected or set()
        for request in self.requests.values():
            if request.request_id in reset_selected:
                request.age = 0
            elif not request.finished:
                request.age += 1

    def _reserve_with_cache_evict(
        self,
        token_count: int,
        *,
        prefix_span,
    ):
        reservation = self.page_pool.reserve_for_tokens(token_count, prefix_span=prefix_span)
        if reservation is not None:
            return reservation
        needed_private_pages = self.page_pool.required_private_pages(
            token_count,
            prefix_span=prefix_span,
        )
        missing_pages = max(0, needed_private_pages - self.page_pool.free_pages)
        if missing_pages == 0:
            return None
        missing_tokens = missing_pages * self.config.page_size
        if self.prefix_store.size_info.evictable_size < missing_tokens:
            return None
        evicted_spans = self.prefix_store.evict(missing_tokens)
        for span in evicted_spans:
            self.page_pool.evict_shared(span)
        return self.page_pool.reserve_for_tokens(token_count, prefix_span=prefix_span)


def start_runtime_process(config: ServerConfig, ingress: mp.Queue, tokenizer_queue: mp.Queue) -> mp.Process:
    process = mp.Process(
        target=SchedulerRuntime(config, ingress, tokenizer_queue).run,
        name="radixinfer-runtime",
        daemon=True,
    )
    process.start()
    return process
