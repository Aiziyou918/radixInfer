from __future__ import annotations

import bisect
import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from radixinfer.core import Batch, Req
    from radixinfer.engine.attention import BaseAttnBackend
    from radixinfer.models.base import BaseLLMModel


def mem_GB(size: int) -> str:
    return f"{size / (1024 ** 3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


# ---------------------------------------------------------------------------
# Shared GPU buffers for graph capture / replay
# ---------------------------------------------------------------------------

@dataclass
class GraphCaptureBuffer:
    """Pre-allocated GPU tensors that are reused across all captured batch sizes.

    During capture the batch's tensor pointers are rewritten to point into
    these buffers so the graph records operations on fixed memory addresses.
    During replay we first copy live data into the buffers, then fire the graph.
    """

    input_ids: torch.Tensor   # int32  [max_bs]
    out_loc: torch.Tensor     # int32  [max_bs]
    positions: torch.Tensor   # int32  [max_bs]
    logits: torch.Tensor      # float32 [max_bs, vocab_size]

    @classmethod
    def allocate(cls, max_bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return cls(
            input_ids=torch.zeros(max_bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
            logits=torch.empty(max_bs, vocab_size, dtype=torch.float32, device=device),
        )

    def bind_to_batch(self, batch: Batch) -> None:
        """Make the batch's tensor fields point into this buffer (capture side)."""
        s = slice(batch.padded_size)
        batch.input_ids = self.input_ids[s]
        batch.out_loc = self.out_loc[s]
        batch.positions = self.positions[s]

    def upload_from_batch(self, batch: Batch) -> None:
        """Copy live batch tensors into the buffer (replay side)."""
        s = slice(batch.padded_size)
        self.input_ids[s].copy_(batch.input_ids)
        self.out_loc[s].copy_(batch.out_loc)
        self.positions[s].copy_(batch.positions)


# ---------------------------------------------------------------------------
# Batch-size schedule
# ---------------------------------------------------------------------------

def _build_bs_schedule(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    max_running_req: int,
) -> List[int]:
    """Return a sorted list of decode batch sizes to capture.

    The schedule always covers ``[1, 2, 4, 8, 16, …]`` up to ``max_bs``.
    ``max_bs`` is chosen to be at least ``max_running_req`` so that a fully
    loaded decode batch can always use the graph path, regardless of GPU size.
    If the caller supplies an explicit list it is used as-is.
    """
    if cuda_graph_bs is not None:
        return sorted(cuda_graph_bs)

    free_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        hw_default = 256 if free_gb > 80 else 160
        cuda_graph_max_bs = max(max_running_req, hw_default)

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------

class GraphRunner:
    """Captures and replays CUDA graphs for the decode phase.

    Graphs are captured from largest to smallest batch size so that later
    (smaller) captures can reuse the CUDA memory pool from the first capture,
    keeping peak memory low.  A ``bisect`` lookup replaces the linear scan
    used by some implementations, making ``pad_batch`` O(log n).
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        max_running_req: int = 256,
    ) -> None:
        from radixinfer.distributed import try_get_tp_info

        tp_info = try_get_tp_info()
        self._is_primary = tp_info.is_primary() if tp_info else True

        bs_schedule = _build_bs_schedule(
            cuda_graph_bs, cuda_graph_max_bs, free_memory, max_running_req
        )
        self.attn_backend = attn_backend
        self.graph_bs_list = bs_schedule          # sorted ascending
        self.max_graph_bs = bs_schedule[-1] if bs_schedule else 0
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device

        self._capture(max_seq_len, vocab_size, model)

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    def _capture(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel) -> None:
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}

        if self.max_graph_bs == 0:
            if self._is_primary:
                print("[radixinfer] CUDA graph capture disabled (max_bs=0).")
            return

        self.attn_backend.init_capture_graph(
            max_seq_len=max_seq_len, bs_list=self.graph_bs_list
        )
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        if self._is_primary:
            print(
                f"[radixinfer] Capturing {len(self.graph_bs_list)} CUDA graphs "
                f"(bs ∈ {self.graph_bs_list[:3]}…{self.graph_bs_list[-3:]}, "
                f"max_bs={self.max_graph_bs})"
            )

        self.buffer = GraphCaptureBuffer.allocate(self.max_graph_bs, vocab_size, self.device)
        self._run_capture(model)

        if self._is_primary:
            print(
                f"[radixinfer] CUDA graphs ready. "
                f"Free memory: {mem_GB(get_free_memory(self.device))}"
            )

    def _run_capture(self, model: BaseLLMModel) -> None:
        """Iterate batch sizes largest-first, capturing one graph per size."""
        from radixinfer.core import Batch, get_global_ctx

        pbar = tqdm(
            reversed(self.graph_bs_list),
            total=len(self.graph_bs_list),
            desc="Graph capture",
            unit="graph",
            disable=not self._is_primary,
        )
        pool = None
        for bs in pbar:
            avail = mem_GB(get_free_memory(self.device))
            pbar.set_postfix_str(f"bs={bs:<4} avail={avail}")

            graph = torch.cuda.CUDAGraph()
            batch = self._make_dummy_batch(bs)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.bind_to_batch(batch)

            with get_global_ctx().forward_batch(batch):
                # Warmup run: fills caches and avoids first-run overhead in the graph.
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()

            if pool is None:
                pool = graph.pool()   # share pool → reuse CUDA graph memory
            self.graph_map[bs] = graph

    def _make_dummy_batch(self, bs: int) -> Batch:
        from radixinfer.core import Batch

        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.padded_reqs = list(batch.reqs)
        return batch

    # ------------------------------------------------------------------
    # Runtime graph selection & replay
    # ------------------------------------------------------------------

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def pad_batch(self, batch: Batch) -> None:
        """Extend batch.padded_reqs to the next captured batch size using bisect."""
        if self.can_use_cuda_graph(batch):
            idx = bisect.bisect_left(self.graph_bs_list, batch.size)
            padded_size = self.graph_bs_list[idx]
        else:
            padded_size = batch.size
        batch.padded_reqs = list(batch.reqs) + [self.dummy_req] * (padded_size - batch.size)

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.upload_from_batch(batch)
        self.attn_backend.prepare_for_replay(batch)
        self.graph_map[batch.padded_size].replay()
        return self.buffer.logits[: batch.size]

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def destroy_cuda_graphs(self) -> None:
        # Must be called before freeing NCCL/pynccl resources to avoid hangs.
        del self.graph_map
        gc.collect()
