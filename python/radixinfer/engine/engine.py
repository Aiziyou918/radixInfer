from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch

from radixinfer.core import Batch, Context, Req, set_global_ctx
from radixinfer.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from radixinfer.engine.attention import create_attention_backend
from radixinfer.engine.config import EngineConfig
from radixinfer.engine.graph import GraphRunner, get_free_memory, mem_GB
from radixinfer.engine.sample import BatchSamplingArgs, Sampler
from radixinfer.layers import set_rope_device
from radixinfer.models import create_model, load_weight
from radixinfer.utils import div_even, torch_dtype


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class CacheAllocation(NamedTuple):
    """Result of KV-cache memory planning."""
    num_pages: int
    num_tokens: int   # num_pages * page_size
    kv_bytes: int     # actual GPU bytes consumed


class MemoryImbalanceError(RuntimeError):
    """Free GPU memory differs by more than 2 GiB across TP ranks."""


def _align_up_32(n: int) -> int:
    return (n + 31) // 32 * 32


class Engine:
    """Single-rank inference engine: model + KV cache + CUDA graphs.

    Initialisation is broken into explicit phases so each resource is easy
    to trace, profile, or mock in tests.  Call ``shutdown()`` when done to
    release GPU and distributed resources in the right order.
    """

    def __init__(self, config: EngineConfig) -> None:
        assert not torch.cuda.is_initialized(), (
            "CUDA must not be initialised before Engine.__init__. "
            "Move all torch.cuda calls inside Engine or its workers."
        )
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        # ── Phase 1: device & stream ─────────────────────────────────────
        gpu_id = config.device_id if config.device_id is not None else config.tp_info.rank
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # ── Phase 2: global inference context ────────────────────────────
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # ── Phase 3: inter-rank communication ────────────────────────────
        self._tp_cpu_group = self._setup_communication(config)
        pre_load_free = self._query_free_memory()
        self._log(config, f"Free memory before model load: {mem_GB(pre_load_free)}")

        # ── Phase 4: model weights ───────────────────────────────────────
        self._setup_model(config)

        # ── Phase 5: KV cache & page table ──────────────────────────────
        alloc = self._setup_kvcache(config, pre_load_free)
        self._log(
            config,
            f"KV cache: {alloc.num_tokens} tokens, "
            f"{mem_GB(alloc.kv_bytes)} "
            f"({alloc.num_pages} pages × {config.page_size} tokens/page)",
        )

        # ── Phase 6: attention & MoE backends ───────────────────────────
        self._setup_backends(config)

        # ── Phase 7: sampler ────────────────────────────────────────────
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_init_free = self._query_free_memory(reset_peak=False)
        self._log(config, f"Free memory after initialisation: {mem_GB(post_init_free)}")

        # ── Phase 8: CUDA graph capture ─────────────────────────────────
        self._setup_graph_runner(config, pre_load_free, alloc)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    @staticmethod
    def _log(config: EngineConfig, msg: str) -> None:
        if config.tp_info.is_primary():
            print(f"[radixinfer] {msg}")

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _setup_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """Initialise the distributed process group and optionally pynccl."""
        timeout = timedelta(seconds=config.distributed_timeout)
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timeout,
                init_method=config.distributed_addr,
            )
            cpu_group = torch.distributed.group.WORLD
            assert cpu_group is not None
            # pynccl buffer covers the largest possible activation tensor.
            max_bytes = (
                config.max_forward_len
                * config.model_config.hidden_size
                * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timeout,
                init_method=config.distributed_addr,
            )
            cpu_group = torch.distributed.new_group(backend="gloo")
            assert cpu_group is not None
        return cpu_group

    def _setup_model(self, config: EngineConfig) -> None:
        """Allocate model on meta device then materialise weights."""
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weights(config))

    def _load_weights(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            # Random initialisation for benchmarking / CI without real checkpoints.
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _setup_kvcache(self, config: EngineConfig, pre_load_free: int) -> CacheAllocation:
        """Allocate the paged KV cache and the global page-address table."""
        from radixinfer.cache.kv_pool import MHAKVCache

        alloc = self._plan_cache(config, pre_load_free)
        self.num_pages = alloc.num_pages

        kv_cache = MHAKVCache(
            num_kv_heads=config.model_config.num_kv_heads,
            num_layers=config.model_config.num_layers,
            head_dim=config.model_config.head_dim,
            num_pages=alloc.num_pages + 1,   # +1 for the dummy/out-of-range page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.ctx.kv_cache = self.kv_cache = kv_cache

        # Page table: one row per request slot (+ 1 dummy), width = aligned max seq len.
        # Every entry stores a raw token position (page_size == 1) or a page index.
        self.max_seq_len = min(config.max_seq_len, alloc.num_tokens)
        self._aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, self._aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        return alloc

    def _plan_cache(self, config: EngineConfig, pre_load_free: int) -> CacheAllocation:
        """Decide how many KV pages to allocate, returning a CacheAllocation."""
        post_load_free = self._query_free_memory()
        bytes_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )

        if config.num_page_override is not None:
            num_pages = config.num_page_override
        else:
            model_bytes = pre_load_free - post_load_free
            available = int(config.memory_ratio * pre_load_free) - model_bytes
            num_pages = available // bytes_per_page

        if num_pages <= 1:
            raise RuntimeError(
                f"Not enough GPU memory for KV cache (computed {num_pages} pages). "
                "Try --memory-ratio / --num-pages, or reduce --max-running-req."
            )
        return CacheAllocation(
            num_pages=num_pages,
            num_tokens=num_pages * config.page_size,
            kv_bytes=num_pages * bytes_per_page,
        )

    def _setup_backends(self, config: EngineConfig) -> None:
        """Attach attention and (optionally) MoE backends to the context."""
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            from radixinfer.moe import create_moe_backend

            moe_name = config.moe_backend if config.moe_backend != "auto" else "fused"
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(moe_name)

    def _setup_graph_runner(
        self,
        config: EngineConfig,
        pre_load_free: int,
        alloc: CacheAllocation,
    ) -> None:
        """Create the dummy request, wire the dummy page slot, capture graphs."""
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
        )
        # Dummy slot points just past valid pages — reads return zeros via padding.
        self.page_table[self.dummy_req.table_idx].fill_(alloc.num_tokens)

        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=pre_load_free,
            max_seq_len=self._aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            max_running_req=config.max_running_req,
        )

    # ------------------------------------------------------------------
    # Memory snapshot
    # ------------------------------------------------------------------

    def _query_free_memory(self, *, reset_peak: bool = True) -> int:
        """Sync, flush cache, then return the worst-case free memory across TP ranks.

        Uses a single all_reduce over [free, -free] to get both min and max
        in one collective.  Raises ``MemoryImbalanceError`` if ranks diverge
        by more than 2 GiB (typically indicates mis-matched model sharding or
        a different GPU variant in the pool).
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        local_free = get_free_memory(self.device)

        probe = torch.tensor([local_free, -local_free], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            probe, op=torch.distributed.ReduceOp.MIN, group=self._tp_cpu_group
        )
        min_free = int(probe[0].item())
        max_free = -int(probe[1].item())

        if max_free - min_free > 2 * 1024 ** 3:
            raise MemoryImbalanceError(
                f"GPU memory is severely imbalanced across TP ranks "
                f"(min={mem_GB(min_free)}, max={mem_GB(max_free)}). "
                "Verify that all ranks use the same GPU model and that no "
                "other process has occupied VRAM on one of the devices."
            )
        # Return the minimum — we must size the cache to the tightest rank.
        return min_free

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream, (
            "forward_batch must run on the engine's dedicated CUDA stream"
        )
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # Async D→H copy; caller waits on copy_done_event before reading CPU tensor.
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        # CUDA graphs must be destroyed before NCCL/pynccl resources are freed;
        # otherwise the process can hang waiting on pending collective ops.
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


# ------------------------------------------------------------------
# Config post-processing
# ------------------------------------------------------------------

def _adjust_config(config: EngineConfig) -> None:
    """Fill in 'auto' fields after hardware detection."""

    def _set(attr: str, value: Any) -> None:
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        from radixinfer.utils.arch import is_sm90_supported, is_sm100_supported

        if is_sm100_supported():
            # Blackwell: FA4 handles both prefill and decode well.
            backend = "fa"
        elif is_sm90_supported():
            # Hopper: FA3 for prefill (variable length), FI for decode (fixed shape).
            backend = "fa,fi"
        else:
            # Pre-Hopper: FlashInfer only (FA3 not supported).
            backend = "fi"
        _set("attention_backend", backend)
        if config.tp_info.is_primary():
            print(f"[radixinfer] Auto attention backend: {backend}")

    if config.moe_backend == "auto":
        _set("moe_backend", "fused")
