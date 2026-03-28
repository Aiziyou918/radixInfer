from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch

if TYPE_CHECKING:
    from radixinfer.core import Batch
    from radixinfer.models.config import ModelConfig


# ---------------------------------------------------------------------------
# Base abstractions
# ---------------------------------------------------------------------------

@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    """Routes prefill/decode to different specialized backends."""

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)


# ---------------------------------------------------------------------------
# Capture data (CUDA Graph buffers)
# ---------------------------------------------------------------------------

@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# FlashAttention backend
# ---------------------------------------------------------------------------

@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int
    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from radixinfer.core import get_global_ctx

        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim ** -0.5
        try:
            from radixinfer.utils.arch import is_sm100_supported
            self.version = 4 if is_sm100_supported() else 3
        except ImportError:
            self.version = 3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        from radixinfer.core import get_global_ctx

        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        from radixinfer.core import get_global_ctx

        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS).to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q = cu_seqlens_k
        else:
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(0)
            cu_seqlens_q = cu_seqlens_q.to(device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None
        max_bs = max(bs_list)
        self.capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        c = self.capture
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=c.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=c.cu_seqlens_q[: bs + 1],
            cache_seqlens=c.seq_lens[:bs],
            max_seqlen_k=c.page_table.size(1) * self.page_size,
            max_seqlen_q=1,
            page_table=c.page_table[:bs, :],
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata) and self.capture is not None
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa_impl(
    q, k_cache, v_cache, page_table, cache_seqlens,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, version,
):
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=True,
            ver=version,
        )
    except ImportError as e:
        raise ImportError(
            "sgl_kernel is required for FlashAttentionBackend. "
            "Install with: pip install sgl-kernel"
        ) from e


# ---------------------------------------------------------------------------
# FlashInfer backend
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << math.ceil(math.log2(n))


@dataclass
class FICaptureData(BaseCaptureData):
    indices: torch.Tensor  # flat 1D page indices for graph replay

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
    wrapper: object  # BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
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
    def _use_tensor_cores(config: ModelConfig) -> bool:
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = self.kvcache.k_cache(layer_id).view(-1, 1, self.kvcache.k_cache(layer_id).shape[2], self.kvcache.k_cache(layer_id).shape[3])
        v_cache = self.kvcache.v_cache(layer_id).view(-1, 1, self.kvcache.v_cache(layer_id).shape[2], self.kvcache.v_cache(layer_id).shape[3])
        return metadata.wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))

    def prepare_metadata(self, batch: Batch) -> None:
        from radixinfer.core import get_global_ctx

        reqs = batch.padded_reqs
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(0)
        if max_seqlen_q == 1:
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(0)

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

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        c = self.capture
        gw = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self._use_tensor_cores(self.config),
            indptr_buffer=c.cu_seqlens_k[: bs + 1],
            indices_buffer=c.indices,
            last_page_len_buffer=c.one_tensor[:bs],
        )
        gw._backend = "fa2"
        gw._int_workspace_buffer = self.int_workspace_buffer
        self.graph_wrappers[bs] = gw
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = gw
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)


# ---------------------------------------------------------------------------
# Backend registry and factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: Dict[str, type] = {
    "fa": FlashAttentionBackend,
    "fi": FlashInferBackend,
}


def validate_attn_backend(backend: str, allow_auto: bool = True) -> str:
    if backend == "auto":
        assert allow_auto, "auto is not allowed here"
        return backend
    for b in (backend.split(",") if "," in backend else [backend]):
        if b not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unsupported attention backend '{b}'. Supported: {list(_BACKEND_REGISTRY.keys())}"
            )
    return backend


def create_attention_backend(backend: str, config: ModelConfig) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma allowed in hybrid backend spec"
        p_name, d_name = backend.split(",", 1)
        if p_name != d_name:
            p_backend = create_attention_backend(p_name, config)
            d_backend = create_attention_backend(d_name, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_name
    return _BACKEND_REGISTRY[backend](config)


# ---------------------------------------------------------------------------
# HuggingFace fallback attention (used by HuggingFaceEngine for testing)
# ---------------------------------------------------------------------------

@dataclass
class PagedAttentionPlan:
    """Paged attention metadata for a single request."""
    table_slot: int
    token_count: int
    write_position: int
    page_ids: tuple
    page_indices: tuple
    kv_page_indices: tuple
    last_page_len: int


@dataclass
class PreparedRequest:
    input_ids: torch.Tensor
    past_key_values: object  # HF-format past_key_values tuple or None
    paged_plan: PagedAttentionPlan | None = None


class PagedAttentionBackend:
    """Minimal paged-attention backend used by test infrastructure."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size

    def prepare_batch(self, token_ids: list, kv_caches: list, metadata) -> list:
        result = []
        for i, tokens in enumerate(token_ids):
            plan = None
            if metadata is not None and i < len(metadata.request_paged_states):
                ps = metadata.request_paged_states[i]
                plan = PagedAttentionPlan(
                    table_slot=ps.table_slot,
                    token_count=ps.token_count,
                    write_position=ps.write_position,
                    page_ids=ps.page_ids,
                    page_indices=ps.page_indices,
                    kv_page_indices=ps.kv_page_indices,
                    last_page_len=ps.last_page_len,
                )
            input_ids = torch.tensor(tokens, dtype=torch.long)
            result.append(PreparedRequest(input_ids=input_ids, past_key_values=None, paged_plan=plan))
        return result


class HuggingFaceFallbackAttentionBackend(PagedAttentionBackend):
    """Bridges HuggingFaceEngine with the paged-KV-cache test infrastructure.

    This is NOT a BaseAttnBackend.  HuggingFace models handle attention
    internally; this class converts between radixInfer's KVCacheView
    format and HuggingFace's past_key_values format, and also populates
    paged attention plans from batch metadata.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(page_size=page_size)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

    def prepare_batch(self, token_ids: list, kv_caches: list, metadata) -> list:
        prepared = super().prepare_batch(token_ids, kv_caches, metadata)
        for i, req in enumerate(prepared):
            kv = kv_caches[i] if i < len(kv_caches) else None
            past_kv = None
            if kv is not None and kv.keys.shape[2] == self.num_heads and kv.keys.shape[3] == self.head_dim:
                # kv.keys: (num_layers, pages*page_size, kv_heads, head_dim)
                # HF expects: tuple[(key, val)] per layer, each (1, heads, seq, head_dim)
                t = kv.token_count
                past_kv = tuple(
                    (
                        kv.keys[l, :t].transpose(0, 1).unsqueeze(0).to(self.device),
                        kv.values[l, :t].transpose(0, 1).unsqueeze(0).to(self.device),
                    )
                    for l in range(kv.keys.shape[0])
                )
            req.past_key_values = past_kv
            req.input_ids = req.input_ids.unsqueeze(0).to(self.device)
        return prepared

    def extract_cache_writes(self, outputs: list, token_counts: list) -> list:
        """Extract K/V from HF model outputs as AttentionCacheWrite objects."""
        from radixinfer.engine.base import AttentionCacheWrite

        result = []
        for output, count in zip(outputs, token_counts):
            pkv = output.past_key_values  # tuple[(key, val)] per layer
            # key per layer: (1, num_heads, seq_len, head_dim)
            # → (seq_len, num_heads, head_dim) → stack → (num_layers, seq_len, ...)
            keys = torch.stack(
                [pkv[l][0].squeeze(0).transpose(0, 1) for l in range(len(pkv))]
            )
            values = torch.stack(
                [pkv[l][1].squeeze(0).transpose(0, 1) for l in range(len(pkv))]
            )
            result.append(
                AttentionCacheWrite(
                    keys=keys[:, :count],
                    values=values[:, :count],
                    token_count=count,
                )
            )
        return result


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "HybridBackend",
    "FAMetadata",
    "FlashAttentionBackend",
    "FIMetadata",
    "FlashInferBackend",
    "create_attention_backend",
    "validate_attn_backend",
    "BaseCaptureData",
    "PreparedRequest",
    "PagedAttentionPlan",
    "PagedAttentionBackend",
    "HuggingFaceFallbackAttentionBackend",
]
