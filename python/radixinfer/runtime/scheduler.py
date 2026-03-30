from __future__ import annotations

import multiprocessing as mp
from typing import Any, NamedTuple, NoReturn, Optional, Set, Tuple, TypeAlias

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
        SchedulerIOMixin.__init__(self, config, tp_cpu_group=tp_cpu_group or engine.tp_cpu_group)

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
            if msg is None:
                raise SystemExit(0)
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
        self.sync_all_ranks()
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
# Process entry-points — mirrors mini-sglang's launch.py pattern
# ---------------------------------------------------------------------------

def _run_scheduler_process(
    config: Any,
    ack_queue: Any = None,
    rank: int = 0,
) -> None:
    """Entry point for each TP-rank subprocess.

    Mirrors mini-sglang's _run_scheduler: construct Scheduler directly,
    barrier, ack, then run_forever().  No SchedulerRuntime bridge needed —
    the Scheduler manages its own ZMQ I/O for both TP=1 and TP>1.
    """
    from radixinfer.config import server_config_to_scheduler_config

    sched_cfg = server_config_to_scheduler_config(config, rank=rank)
    sched = Scheduler(sched_cfg)

    # CPU-side barrier: all ranks must finish ZMQ socket setup before rank 0
    # signals readiness.  Without this, rank 0 may publish before rank 1's
    # sub-socket has connected, silently dropping the first batch of messages.
    sched.sync_all_ranks()

    if ack_queue is not None:
        try:
            ack_queue.put("Scheduler is ready")
        except Exception:
            pass

    try:
        sched.run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        sched.shutdown()


def start_runtime_process(
    config: Any,
    *,
    ack_queue: Any = None,
) -> list:
    """Spawn one daemon subprocess per TP rank running the Scheduler loop.

    Returns a list of mp.Process objects (one per TP rank).
    Only rank 0 receives ack_queue; its ack implies all ranks have completed
    torch.distributed.init_process_group and ZMQ setup.
    """
    tp_size = getattr(config, "tp_size", 1)
    processes = []
    for rank in range(tp_size):
        process = mp.Process(
            target=_run_scheduler_process,
            args=(config, ack_queue if rank == 0 else None, rank),
            name=f"radixinfer-runtime-{rank}",
            daemon=True,
        )
        process.start()
        processes.append(process)
    return processes
