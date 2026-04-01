from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal

import torch

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from radixinfer.core import Batch
    from radixinfer.models.config import ModelConfig


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << math.ceil(math.log2(n))


@dataclass
class FICaptureData(BaseCaptureData):
    indices: torch.Tensor

    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens


@dataclass
class FIMetadata(BaseAttnMetadata):
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor
    last_page_len_cpu: torch.Tensor
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: Literal[1]
    pos_encoding_mode: str
    seq_lens_cpu: torch.Tensor
    dtype: torch.dtype
    wrapper: object
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: "ModelConfig") -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )
        from radixinfer.core import get_global_ctx
        from radixinfer.distributed import get_tp_info
        from radixinfer.utils import div_even

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device

        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer, kv_layout="NHD", backend="fa2"
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self._use_tensor_cores(config),
            kv_layout="NHD",
            backend="fa2",
        )
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrapper._int_workspace_buffer = self.int_workspace_buffer

        tp_size = get_tp_info().size
        self.qo_head_local = div_even(config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu = torch.tensor([], dtype=torch.int32, pin_memory=True)
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, object] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    @staticmethod
    def _use_tensor_cores(config: "ModelConfig") -> bool:
        return (config.num_qo_heads // config.num_kv_heads) >= 4

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: "Batch"
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = self.kvcache.k_cache(layer_id).view(
            -1,
            1,
            self.kvcache.k_cache(layer_id).shape[2],
            self.kvcache.k_cache(layer_id).shape[3],
        )
        v_cache = self.kvcache.v_cache(layer_id).view(
            -1,
            1,
            self.kvcache.v_cache(layer_id).shape[2],
            self.kvcache.v_cache(layer_id).shape[3],
        )
        return metadata.wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))

    def prepare_metadata(self, batch: "Batch") -> None:
        from radixinfer.core import get_global_ctx

        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(0)
        if max_seqlen_q == 1:
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **cpu_kwargs)
        elif all(length == 0 for length in cached_lens):
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(self.device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrapper if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None
        max_bs = max(bs_list)
        indices = torch.zeros(max_bs * max_seq_len, dtype=torch.int32, device=self.kvcache.device)
        self.capture = FICaptureData(
            seq_lens=torch.ones(max_bs, dtype=torch.int32, device=self.kvcache.device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=self.kvcache.device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.kvcache.device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.kvcache.device),
            page_table=indices,
            indices=indices,
        )
        self.max_graph_bs = max_bs
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: "Batch") -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        graph_wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self._use_tensor_cores(self.config),
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        graph_wrapper._backend = "fa2"
        graph_wrapper._int_workspace_buffer = self.int_workspace_buffer
        self.graph_wrappers[bs] = graph_wrapper
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = graph_wrapper
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: "Batch") -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
