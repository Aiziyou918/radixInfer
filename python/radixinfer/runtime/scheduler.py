from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, List, NamedTuple, NoReturn, Optional, Set, Tuple, TypeAlias

import torch

from radixinfer.core import Batch, Req
from radixinfer.engine.engine import Engine, ForwardOutput
from radixinfer.engine.sample import BatchSamplingArgs
from radixinfer.runtime.cache_manager import CacheManager
from radixinfer.runtime.decode import DecodeManager
from radixinfer.runtime.io import SchedulerIOMixin
from radixinfer.runtime.prefill import ChunkedReq, PrefillManager
from radixinfer.runtime.table import TableManager
from radixinfer.transport.protocol import DetokenizeRequest
from radixinfer.transport.queues import make_zmq_pull, make_zmq_push

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D
    write_tuple: Indice2D


ForwardData: TypeAlias = "tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """Overlap-scheduled inference scheduler, aligned with mini-sglang Scheduler.

    Inherits from SchedulerIOMixin which wires up ZMQ-based (or queue-based in
    offline mode) receive_msg() / send_result() methods.

    Key responsibilities:
    - Receive tokenized requests via receive_msg()
    - Schedule prefill / decode batches
    - Overlap GPU execution with batch post-processing (CPU-side)
    - Send detokenize results back via send_result()
    """

    def __init__(self, config, engine: Engine | None = None, tp_cpu_group=None):
        if engine is None:
            engine = Engine(config)

        self.engine = engine
        self.device = engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(engine.stream)
        self._config = config
        torch.cuda.set_stream(self.stream)

        self.table_manager = TableManager(config.max_running_req, engine.page_table)
        self.cache_manager = CacheManager(
            engine.num_pages,
            config.page_size,
            engine.page_table,
            getattr(config, "cache_type", "radix"),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.prefill_budget = getattr(config, "max_extend_tokens", 8192)
        self.token_pool = self.table_manager.token_pool

        # EOS tracking from tokenizer
        self.eos_token_id: int = 0

        # Set up I/O via mixin (ZMQ for production, queue.Queue for offline/tests)
        SchedulerIOMixin.__init__(self, config, tp_cpu_group=tp_cpu_group)

    def enqueue(self, request) -> None:
        """Put a tokenized request into the offline input queue (offline mode only)."""
        self._in_queue.put(request)

    def dequeue_results(self) -> list:
        """Drain all available detokenize results."""
        results = []
        while not self._out_queue.empty():
            results.append(self._out_queue.get_nowait())
        return results

    def run_when_idle(self) -> None:
        """Called while blocking-waiting for a new message (overrides mixin hook)."""
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # Mirror mini-sglang's two-branch design:
        #
        # non-overlap branch:
        #   Enter engine_stream_ctx once and hold it for the entire loop so
        #   that every normal_loop() → _forward() → engine.forward_batch()
        #   call sees torch.cuda.current_stream() == engine.stream.
        #   wait_stream ensures the scheduler stream has caught up before the
        #   engine stream starts executing.
        #
        # overlap branch:
        #   overlap_loop() manages engine_stream_ctx itself per-iteration.
        #   We only assert the entry condition so bugs are caught early.
        if getattr(self._config, "disable_overlap_scheduling", False):
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream, (
                "run_forever (overlap mode) must be entered while the scheduler "
                "stream is current; call torch.cuda.set_stream(self.stream) first."
            )
            data = None
            while True:
                data = self.overlap_loop(data)

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        blocking = not (
            last_data is not None
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        self._drain_input_queue(blocking=blocking)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        self._drain_input_queue(blocking=blocking)

        forward_input = self._schedule_next_batch()
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))
            self._process_last_data(ongoing_data)

    def _drain_input_queue(self, blocking: bool) -> None:
        msgs = self.receive_msg(blocking=blocking)
        for msg in msgs:
            self._process_new_req(msg)

    def _process_new_req(self, req) -> None:
        from radixinfer.transport.protocol import AbortRequest, TokenizedRequest

        if isinstance(req, TokenizedRequest) or hasattr(req, "token_ids"):
            self.prefill_manager.add_one_req(req)
        elif isinstance(req, AbortRequest) or hasattr(req, "request_id") and not hasattr(req, "token_ids"):
            uid = getattr(req, "request_id", None)
            if uid is not None:
                self.prefill_manager.abort_req(uid)
                self.decode_manager.abort_req(uid)

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        forward_input, (_, next_tokens_cpu, copy_done) = last_data[0], last_data[1]
        batch = forward_input.batch
        copy_done.synchronize()

        new_finished: Set[Req] = set()
        results: list[DetokenizeRequest] = []
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token_id = int(next_token.item())

                stop_ids = getattr(req.sampling_params, "stop_token_ids", [])
                eos_hit = (
                    not req.sampling_params.ignore_eos
                    and bool(stop_ids)
                    and next_token_id in stop_ids
                )
                finished = (not req.can_decode) or eos_hit
                finish_reason = "stop" if eos_hit else ("length" if not req.can_decode else "running")

                prompt_len = len(req.input_ids) - req.output_len
                completion_len = req.output_len - req.remain_len

                results.append(DetokenizeRequest(
                    request_id=req.uid,
                    token_id=next_token_id,
                    finished=finished,
                    finish_reason=finish_reason,
                    emit_text=True,
                    prompt_tokens=max(0, prompt_len),
                    completion_tokens=max(0, completion_len),
                ))

                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished
        self.send_result(results)

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _schedule_next_batch(self) -> ForwardInput | None:
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        return self._prepare_batch(batch)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.engine.shutdown()


# ---------------------------------------------------------------------------
# Hot-path batch metadata builders (pin_memory + non_blocking H2D)
# ---------------------------------------------------------------------------

def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return (
        mapping_host.to(device, non_blocking=True),
        batch.positions.to(torch.int64),
    )


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return (
        mapping_host.to(device, non_blocking=True),
        write_host.to(device, non_blocking=True),
    )


# ---------------------------------------------------------------------------
# _SimpleInlineScheduler — lightweight debug scheduler (no CUDA required)
# ---------------------------------------------------------------------------

class _SimpleInlineScheduler:
    """Single-process scheduler using the legacy DummyEngine interface.

    Used when CUDA is unavailable or when engine_kind='dummy'.  Processes one
    token per request per tick.  No page cache, no batching — purely for debug
    and integration testing.
    """

    def __init__(self, output_queue: Any) -> None:
        from radixinfer.engine.dummy import DummyEngine

        self._engine = DummyEngine()
        self._output = output_queue
        self._pending: dict[int, dict] = {}

    def enqueue(self, req: Any) -> None:
        if hasattr(req, "token_ids"):
            self._pending[req.request_id] = {
                "token_ids": list(req.token_ids),
                "generated": [],
                "sampling": req.sampling,
                "stop_ids": list(getattr(req, "stop_token_ids", ())),
                "eos": getattr(req, "eos_token_id", None),
            }

    def normal_loop(self) -> None:
        from radixinfer.engine.base import DecodeInput

        finished_ids = []
        for uid, state in list(self._pending.items()):
            all_ids = state["token_ids"] + state["generated"]
            batch = DecodeInput(request_ids=[uid], token_ids=[all_ids])
            out = self._engine.decode(batch)
            next_token = out.next_token_ids[0]
            state["generated"].append(next_token)

            sampling = state["sampling"]
            stop_ids = list(state["stop_ids"])
            eos = state["eos"]
            if eos is not None:
                stop_ids.append(eos)

            eos_hit = (
                not getattr(sampling, "ignore_eos", False)
                and bool(stop_ids)
                and next_token in stop_ids
            )
            finished = eos_hit or len(state["generated"]) >= sampling.max_tokens
            finish_reason = "stop" if eos_hit else "length"

            self._output.put_nowait(DetokenizeRequest(
                request_id=uid,
                token_id=next_token,
                finished=finished,
                finish_reason=finish_reason,
                emit_text=True,
                prompt_tokens=len(state["token_ids"]),
                completion_tokens=len(state["generated"]),
            ))

            if finished:
                finished_ids.append(uid)

        for uid in finished_ids:
            del self._pending[uid]

    def run_forever(self) -> NoReturn:
        import time

        while True:
            self.normal_loop()
            if not self._pending:
                time.sleep(0.001)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# _DebugCacheManager — wraps PagePool for debug-mode reserve/evict
# ---------------------------------------------------------------------------

class _DebugCacheManager:
    """Minimal cache manager for debug SchedulerRuntime.

    Exposes reserve(token_count, prefix_span) which allocates pages from
    PagePool, evicting prefix cache entries as needed.
    """

    def __init__(self, page_pool: Any, prefix_store: Any, page_size: int) -> None:
        self._page_pool = page_pool
        self._prefix_store = prefix_store
        self._page_size = page_size

    def reserve(self, token_count: int, prefix_span: Any) -> Any:
        needed_pages = (token_count + self._page_size - 1) // self._page_size
        shared_pages = len(prefix_span.page_ids) if prefix_span is not None else 0
        private_needed = max(0, needed_pages - shared_pages)
        evict_size = max(0, (private_needed - self._page_pool.free_pages) * self._page_size)
        if evict_size > 0:
            evicted = self._prefix_store.evict(evict_size)
            for span in evicted:
                self._page_pool.reclaim_span(span)
        return self._page_pool.reserve_for_tokens(token_count, prefix_span=prefix_span)


# ---------------------------------------------------------------------------
# SchedulerRuntime — lightweight wrapper for api/server.py integration
# ---------------------------------------------------------------------------

class SchedulerRuntime:
    """Wraps Scheduler and bridges external queue I/O for api/server.py.

    Automatically selects the implementation:
    - Debug (engine_kind in ("dummy", "hf")): CPU-only runtime with direct
      attribute access used by tests (_run_prefill, _run_decode, etc.).
    - Real GPU: Scheduler backed by the new Engine with full hot path.

    Inline mode: _drain_ingress() + _tick() are called cooperatively.
    Multiprocess mode: run_forever() blocks in a subprocess.
    """

    def __init__(self, config: Any, runtime_ingress: Any, output_queue: Any) -> None:
        engine_kind = getattr(config, "engine_kind", "dummy")
        is_debug = engine_kind in ("dummy", "hf")

        self._ingress = runtime_ingress
        self.tokenizer_queue = output_queue

        if is_debug:
            self._impl: Any = None
            self._setup_debug_mode(config)
        else:
            from radixinfer.config import server_config_to_scheduler_config

            sched_cfg = server_config_to_scheduler_config(config)
            sched = Scheduler(sched_cfg)
            sched._out_queue = output_queue
            self._impl = sched

    def _setup_debug_mode(self, config: Any) -> None:
        from radixinfer.cache.page_pool import PagePool
        from radixinfer.cache.prefix_store import PrefixStore
        from radixinfer.runtime.executor import Executor
        from radixinfer.runtime.table import _DebugTableManager

        page_size = getattr(config, "page_size", 1)
        total_pages = getattr(config, "total_pages", 64)
        kv_cache_dim = getattr(config, "kv_cache_dim", 16)
        kv_num_layers = getattr(config, "kv_num_layers", 2)
        kv_num_heads = getattr(config, "kv_num_heads", 2)
        max_batch_size = getattr(config, "max_batch_size", 32)
        max_prefill_tokens = getattr(config, "max_prefill_tokens", 8192)
        prefix_cache_capacity = getattr(config, "prefix_cache_capacity", 4096)
        engine_kind = getattr(config, "engine_kind", "dummy")

        self.page_pool = PagePool(
            total_pages=total_pages,
            page_size=page_size,
            kv_cache_dim=kv_cache_dim,
            kv_num_layers=kv_num_layers,
            kv_num_heads=kv_num_heads,
        )
        self.prefix_store = PrefixStore(
            capacity=prefix_cache_capacity,
            page_size=page_size,
        )
        self.requests: dict[int, Any] = {}
        self._max_prefill_tokens = max_prefill_tokens
        self._page_size = page_size

        self.table_manager = _DebugTableManager(max_slots=max_batch_size, page_size=page_size)

        if engine_kind == "hf":
            from radixinfer.engine.hf import HuggingFaceEngine

            model = getattr(config, "model", "debug")
            device = getattr(config, "device", "auto")
            self.engine: Any = HuggingFaceEngine(
                model_name=model,
                device=device,
                kv_num_layers=kv_num_layers,
                kv_num_heads=kv_num_heads,
                kv_cache_dim=kv_cache_dim,
                page_size=page_size,
            )
        else:
            from radixinfer.engine.dummy import DummyEngine

            self.engine = DummyEngine()

        self.cache_manager = _DebugCacheManager(self.page_pool, self.prefix_store, page_size)
        self.executor = Executor(page_pool=self.page_pool, table_manager=self.table_manager)

    # ------------------------------------------------------------------
    # Debug-mode scheduling methods
    # ------------------------------------------------------------------

    def _run_prefill(self, request_ids: List[int]) -> None:
        from radixinfer.engine.base import PrefillInput
        from radixinfer.runtime.types import RequestPhase

        for req_id in request_ids:
            req = self.requests[req_id]

            if req.table_slot is None:
                req.table_slot = self.table_manager.allocate()

            # Full prefix hit — nothing to compute
            if req.prefix_matched >= len(req.prompt_tokens):
                req.cache_span = req.prefix_span
                req.phase = RequestPhase.READY_TO_DECODE
                continue

            tokens_this_tick = min(req.remaining_prefill_tokens, self._max_prefill_tokens)
            req.prefill_cursor += tokens_this_tick

            if not req.prefill_complete:
                req.phase = RequestPhase.PREFILLING
                continue

            # Prefill completes this tick
            max_decode = req.sampling.max_tokens if req.sampling is not None else 0
            total_tokens = len(req.prompt_tokens) + max_decode

            if req.reservation is None:
                req.reservation = self.cache_manager.reserve(total_tokens, req.prefix_span)

            if req.prefix_span is not None and req.prefix_matched > 0:
                self.page_pool.share_span(req.prefix_span)

            req.cache_span = self.page_pool.write_tokens(req.reservation, req.prompt_tokens)
            req.prefix_span = req.cache_span

            self.table_manager.materialize_span(
                req.table_slot,
                req.cache_span.page_ids,
                req.prompt_tokens[: req.cache_span.token_count],
            )

            # Engine prefill + KV write
            prefill_input = PrefillInput(
                request_ids=[req_id],
                token_ids=[req.prompt_tokens],
                kv_caches=[None],
            )
            prefill_output = self.engine.prefill(prefill_input)
            if prefill_output.kv_writes:
                kv_write = prefill_output.kv_writes[0]
                if kv_write.token_count > 0 and hasattr(kv_write.keys, "shape") and kv_write.keys.ndim == 4:
                    self.page_pool.write_kv(req.reservation, kv_write.keys, kv_write.values)

            # Insert into prefix store
            req.prefix_cache_key, evicted = self.prefix_store.insert(
                req.prompt_tokens[: req.cache_span.token_count],
                req.cache_span,
            )
            for evicted_span in evicted:
                self.page_pool.reclaim_span(evicted_span)
            if req.prefix_cache_key is not None:
                self.prefix_store.lock(req.prefix_cache_key)

            req.phase = RequestPhase.READY_TO_DECODE

    def _run_decode(self, request_ids: List[int]) -> None:
        from radixinfer.engine.base import DecodeInput
        from radixinfer.runtime.types import RequestPhase
        from radixinfer.transport.protocol import DetokenizeRequest

        for req_id in request_ids:
            req = self.requests[req_id]
            if req.cache_span is None or req.table_slot is None:
                continue

            last_tokens = self.page_pool.read_span(req.cache_span)[-1:]
            prev_count = max(0, req.cached_token_count - 1)
            kv_cache = self.page_pool.read_kv(req.cache_span, token_count=prev_count) if prev_count > 0 else None

            batch = DecodeInput(
                request_ids=[req_id],
                token_ids=[last_tokens],
                kv_caches=[kv_cache],
            )
            output = self.engine.decode(batch)
            next_token_id = output.next_token_ids[0]
            req.generated_tokens.append(next_token_id)

            if req.reservation is not None:
                write_pos = req.cache_span.token_count
                req.cache_span = self.page_pool.write_tokens(
                    req.reservation, [next_token_id], start_offset=write_pos
                )
                kv_writes = getattr(output, "kv_writes", [])
                if kv_writes and kv_writes[0].token_count > 0 and hasattr(kv_writes[0].keys, "shape") and kv_writes[0].keys.ndim == 4:
                    self.page_pool.write_kv(
                        req.reservation, kv_writes[0].keys, kv_writes[0].values, start_offset=write_pos
                    )

            if req.cache_span is not None:
                pos = req.cache_span.token_count - 1
                page_id = req.cache_span.page_ids[pos // self._page_size]
                self.table_manager.append_token(req.table_slot, pos, page_id, next_token_id)

            stop_ids = list(req.stop_token_ids)
            if req.eos_token_id is not None:
                stop_ids.append(req.eos_token_id)
            ignore_eos = getattr(req.sampling, "ignore_eos", False) if req.sampling is not None else False
            eos_hit = bool(stop_ids) and next_token_id in stop_ids and not ignore_eos
            finished = eos_hit or req.remaining_tokens <= 0
            finish_reason = "stop" if eos_hit else ("length" if finished else "running")
            emit_text = not eos_hit

            if finished:
                req.phase = RequestPhase.FINISHED
                if req.prefix_cache_key is not None:
                    self.prefix_store.unlock(req.prefix_cache_key)

            self.tokenizer_queue.put_nowait(DetokenizeRequest(
                request_id=req_id,
                token_id=next_token_id,
                finished=finished,
                finish_reason=finish_reason,
                emit_text=emit_text,
                prompt_tokens=len(req.prompt_tokens),
                completion_tokens=len(req.generated_tokens),
            ))

    def _drain_ingress(self) -> None:
        """Read all pending items from runtime_ingress into the scheduler."""
        while True:
            try:
                req = self._ingress.get_nowait()
            except Exception:
                break
            if req is None:
                break
            if self._impl is not None:
                self._impl.enqueue(req)
            else:
                self._debug_enqueue(req)

    def _debug_enqueue(self, req: Any) -> None:
        from radixinfer.runtime.types import RuntimeRequest

        if hasattr(req, "token_ids"):  # TokenizedRequest
            runtime_req = RuntimeRequest(
                request_id=req.request_id,
                prompt_tokens=list(req.token_ids),
                sampling=req.sampling,
                eos_token_id=getattr(req, "eos_token_id", None),
                stop_token_ids=getattr(req, "stop_token_ids", ()),
            )
            self.requests[req.request_id] = runtime_req
        elif hasattr(req, "request_id") and not hasattr(req, "token_ids"):  # AbortRequest
            req_id = getattr(req, "request_id", None)
            r = self.requests.pop(req_id, None)
            if r is not None:
                # Emit a finished signal so _run_inline_request exits
                self.tokenizer_queue.put_nowait(DetokenizeRequest(
                    request_id=req_id,
                    token_id=0,
                    finished=True,
                    finish_reason="abort",
                    emit_text=False,
                    prompt_tokens=len(r.prompt_tokens),
                    completion_tokens=len(r.generated_tokens),
                ))

    def _tick(self) -> None:
        """Run one scheduling iteration (non-blocking)."""
        if self._impl is not None:
            self._impl.normal_loop()
        else:
            self._debug_tick()

    def _debug_tick(self) -> None:
        from radixinfer.runtime.types import RequestPhase

        prefill_ids = [
            rid for rid, r in list(self.requests.items())
            if r.phase in (RequestPhase.WAIT_PREFILL, RequestPhase.PREFILLING)
        ]
        if prefill_ids:
            self._run_prefill(prefill_ids)

        # Decode requests that are ready (phase stays READY_TO_DECODE until finished)
        decode_ids = [
            rid for rid, r in list(self.requests.items())
            if r.phase == RequestPhase.READY_TO_DECODE
        ]
        if decode_ids:
            self._run_decode(decode_ids)

        finished = [rid for rid, r in list(self.requests.items()) if r.phase == RequestPhase.FINISHED]
        for rid in finished:
            del self.requests[rid]

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """Blocking loop — intended for subprocess use.

        When a real Scheduler is present the entire loop runs inside
        engine_stream_ctx so that every normal_loop() → forward_batch()
        call sees torch.cuda.current_stream() == engine.stream.
        This mirrors mini-sglang's DISABLE_OVERLAP_SCHEDULING branch in
        Scheduler.run_forever() and satisfies the engine assertion.
        """
        if self._impl is not None:
            with self._impl.engine_stream_ctx:
                self._impl.engine.stream.wait_stream(self._impl.stream)
                while True:
                    self._drain_ingress()
                    self._impl.normal_loop()
        else:
            while True:
                self._drain_ingress()
                self._debug_tick()


def _run_scheduler_process(
    config: Any,
    runtime_ingress: Any,
    output_queue: Any,
    ack_queue: Any = None,
) -> None:
    """Entry point for the runtime subprocess."""
    if isinstance(runtime_ingress, str):
        runtime_ingress = make_zmq_pull(runtime_ingress, create=True)
    if isinstance(output_queue, str):
        output_queue = make_zmq_push(output_queue, create=False)
    runtime = SchedulerRuntime(config, runtime_ingress, output_queue)
    # Signal to the parent process that initialisation (model loading) is done
    if ack_queue is not None:
        try:
            ack_queue.put("Scheduler is ready")
        except Exception:
            pass
    runtime.run_forever()


def start_runtime_process(
    config: Any,
    runtime_ingress: Any,
    output_queue: Any,
    *,
    ack_queue: Any = None,
) -> mp.Process:
    """Spawn a daemon subprocess running the Scheduler loop."""
    process = mp.Process(
        target=_run_scheduler_process,
        args=(config, runtime_ingress, output_queue, ack_queue),
        name="radixinfer-runtime",
        daemon=True,
    )
    process.start()
    return process
