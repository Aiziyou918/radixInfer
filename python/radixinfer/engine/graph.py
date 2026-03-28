from __future__ import annotations

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


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, max_bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return cls(
            input_ids=torch.zeros(max_bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
            logits=torch.empty(max_bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        s = slice(batch.padded_size)
        batch.input_ids = self.input_ids[s]
        batch.out_loc = self.out_loc[s]
        batch.positions = self.positions[s]

    def copy_from(self, batch: Batch) -> None:
        s = slice(batch.padded_size)
        self.input_ids[s].copy_(batch.input_ids)
        self.out_loc[s].copy_(batch.out_loc)
        self.positions[s].copy_(batch.positions)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        cuda_graph_max_bs = 256 if free_gb > 80 else 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


class GraphRunner:
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
    ) -> None:
        from radixinfer.distributed import try_get_tp_info

        cuda_graph_bs = _determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory)
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        tp_info = try_get_tp_info()
        self._is_primary = tp_info.is_primary() if tp_info else True
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel) -> None:
        from radixinfer.core import Batch, get_global_ctx

        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            if self._is_primary:
                print("CUDA graph is disabled.")
            return

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        if self._is_primary:
            print(f"Capturing CUDA graphs: {self.graph_bs_list}")
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Capturing CUDA graphs",
            unit="bs",
            disable=not self._is_primary,
        )
        pool = None
        for bs in pbar:
            free_mem = get_free_memory(self.device)
            pbar.set_description(f"Capturing bs={bs:<3} | avail={mem_GB(free_mem)}")
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = list(batch.reqs)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                # warmup
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()
            self.graph_map[bs] = graph

        if self._is_primary:
            print(f"CUDA graphs captured. Free memory: {mem_GB(get_free_memory(self.device))}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        if self.can_use_cuda_graph(batch):
            padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        else:
            padded_size = batch.size
        batch.padded_reqs = list(batch.reqs) + [self.dummy_req] * (padded_size - batch.size)

    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
