from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 1919
    model: str = "debug"
    tokenizer_workers: int = 1
    runtime_workers: int = 1
    max_running_requests: int = 128
    max_prefill_tokens: int = 2048
    page_size: int = 16
    total_pages: int = 4096
    kv_cache_dim: int = 16
    kv_num_layers: int = 2
    kv_num_heads: int = 2
    max_batch_size: int = 32
    engine_kind: Literal["dummy", "hf", "real"] = "hf"
    prefix_cache_capacity: int = 4096
    queue_poll_interval: float = 0.005
    scheduler_tick_interval: float = 0.001
    max_queue_drain: int = 64
    default_max_tokens: int = 64
    tokenizer_name: str | None = None
    device: str = "auto"
    start_method: str = "spawn"
    use_zmq: bool = True
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    _unique_suffix: str = field(default_factory=lambda: f".pid={os.getpid()}")

    @property
    def zmq_tokenizer_addr(self) -> str:
        return "ipc:///tmp/radixinfer_3" + self._unique_suffix

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/radixinfer_4" + self._unique_suffix

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/radixinfer_0" + self._unique_suffix


def server_config_to_scheduler_config(cfg: ServerConfig):
    """Convert a ServerConfig (high-level API config) to a SchedulerConfig
    that the Engine and Scheduler can consume.

    Single-GPU (rank=0, size=1) by default.  dtype is float16 unless the
    model name / device hints suggest otherwise.
    """
    import torch

    from radixinfer.distributed import DistributedInfo
    from radixinfer.runtime.scheduler_config import SchedulerConfig

    use_dummy = cfg.engine_kind in ("dummy",) or cfg.model in ("debug", "dummy")

    return SchedulerConfig(
        model_path=cfg.model,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        page_size=cfg.page_size,
        max_running_req=cfg.max_running_requests,
        num_page_override=cfg.total_pages,
        max_extend_tokens=cfg.max_prefill_tokens,
        use_dummy_weight=use_dummy,
        _unique_suffix=cfg._unique_suffix,
    )
