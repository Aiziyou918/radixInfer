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


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized(), "CUDA must not be initialized before Engine"
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        gpu_id = config.device_id if config.device_id is not None else config.tp_info.rank
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        if config.tp_info.is_primary():
            print(f"Free memory before model load: {mem_GB(init_free_memory)}")

        # Model
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # KV cache
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        from radixinfer.cache.kv_pool import MHAKVCache

        kv_cache = MHAKVCache(
            num_kv_heads=config.model_config.num_kv_heads,
            num_layers=config.model_config.num_layers,
            head_dim=config.model_config.head_dim,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.ctx.kv_cache = self.kv_cache = kv_cache

        # Page table: shape (max_running_req + 1, aligned_max_seq_len)
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # Attention backend
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )

        # MoE backend (only for MoE models)
        if config.model_config.is_moe:
            from radixinfer.moe import create_moe_backend
            moe_backend_name = config.moe_backend if config.moe_backend != "auto" else "fused"
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(moe_backend_name)

        # Sampler
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        if config.tp_info.is_primary():
            print(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # Dummy request for graph padding
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)

        # CUDA Graph
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            max_running_req=config.max_running_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len
                * config.model_config.hidden_size
                * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
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
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Insufficient memory for KV cache. Try --memory-ratio or --num-pages."
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        if config.tp_info.is_primary():
            print(f"KV cache: {num_tokens} tokens, {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free_memory = get_free_memory(self.device)
        mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free = int(mem_tensor[0].item())
        max_free = -int(mem_tensor[1].item())
        if max_free - min_free > 2 * 1024 * 1024 * 1024:
            raise RuntimeError(
                f"Memory imbalanced across TP ranks: "
                f"min={mem_GB(min_free)}, max={mem_GB(max_free)}"
            )
        return min_free, max_free

    @torch.inference_mode()
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _adjust_config(config: EngineConfig) -> None:
    def override(attr: str, value: Any):
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        try:
            from radixinfer.utils.arch import is_sm90_supported, is_sm100_supported

            if is_sm100_supported():
                backend = "fa"  # trtllm not yet integrated
            elif is_sm90_supported():
                backend = "fa,fi"
            else:
                backend = "fi"
        except ImportError:
            backend = "fi"
        override("attention_backend", backend)
        if config.tp_info.is_primary():
            print(f"Auto-selected attention backend: {backend}")

    if config.moe_backend == "auto":
        override("moe_backend", "fused")
