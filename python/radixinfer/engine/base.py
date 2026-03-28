from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from radixinfer.cache.page_pool import KVCacheView


@dataclass(frozen=True)
class DecodeInput:
    request_ids: list[int]
    token_ids: list[list[int]]
    kv_caches: list[KVCacheView] = field(default_factory=list)


@dataclass(frozen=True)
class DecodeOutput:
    next_token_ids: list[int]


class Engine(Protocol):
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        ...
