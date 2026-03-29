from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    max_batch_size: int = 32
    prefix_cache_capacity: int = 4096
    queue_poll_interval: float = 0.005
    scheduler_tick_interval: float = 0.001
    max_queue_drain: int = 64
    default_max_tokens: int = 64
    tokenizer_name: str | None = None
    device: str = "auto"
    tp_size: int = 1
    dist_port: int = 29500
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


def _device_to_rank(device: str) -> int:
    """Parse 'cuda:N' → N, 'auto' / 'cuda' → 0."""
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (IndexError, ValueError):
            pass
    return 0


def server_config_to_scheduler_config(cfg: ServerConfig, rank: int = 0):
    """Convert a ServerConfig (high-level API config) to a SchedulerConfig
    that the Engine and Scheduler can consume.

    rank: TP rank for this process (0 for single-GPU).
    """
    import torch

    from radixinfer.distributed import DistributedInfo
    from radixinfer.runtime.scheduler_config import SchedulerConfig

    base_device_id = _device_to_rank(cfg.device)

    return SchedulerConfig(
        model_path=cfg.model,
        tp_info=DistributedInfo(rank=rank, size=cfg.tp_size),
        device_id=base_device_id + rank,
        dtype=torch.float16,
        # pynccl kernel requires a compiled C extension; fall back to native
        # torch.distributed NCCL when running TP>1 without the extension.
        use_pynccl=(cfg.tp_size == 1),
        page_size=cfg.page_size,
        max_running_req=cfg.max_running_requests,
        num_page_override=cfg.total_pages,
        max_extend_tokens=cfg.max_prefill_tokens,
        dist_port=cfg.dist_port,
        _unique_suffix=cfg._unique_suffix,
        # TP=1: SchedulerRuntime bridges queues manually (offline mode).
        # TP>1: SchedulerIOMixin manages ZMQ sockets directly (online mode).
        offline_mode=(cfg.tp_size == 1),
    )
