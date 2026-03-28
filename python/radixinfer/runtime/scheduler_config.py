from __future__ import annotations

import os
from dataclasses import dataclass, field

from radixinfer.engine.config import EngineConfig


def _get_pid_suffix() -> str:
    return f".pid={os.getpid()}"


def _default_disable_overlap() -> bool:
    # Prefer ENV singleton; fall back to direct os.getenv if ENV not yet available.
    try:
        from radixinfer.env import ENV
        return bool(ENV.DISABLE_OVERLAP_SCHEDULING)
    except Exception:
        return os.getenv("RADIXINFER_DISABLE_OVERLAP_SCHEDULING", "0") != "0"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    # When True, skip overlap scheduling and run normal_loop() synchronously.
    # Controlled by RADIXINFER_DISABLE_OVERLAP_SCHEDULING env var (via ENV singleton).
    disable_overlap_scheduling: bool = field(default_factory=_default_disable_overlap)
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/radixinfer_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/radixinfer_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/radixinfer_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
