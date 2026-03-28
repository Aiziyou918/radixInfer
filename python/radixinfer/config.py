from __future__ import annotations

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
    max_batch_size: int = 32
    engine_kind: Literal["dummy", "hf"] = "hf"
    prefix_cache_capacity: int = 4096
    queue_poll_interval: float = 0.005
    scheduler_tick_interval: float = 0.001
    max_queue_drain: int = 64
    default_max_tokens: int = 64
    tokenizer_name: str | None = None
    device: str = "auto"
    start_method: str = "spawn"
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
