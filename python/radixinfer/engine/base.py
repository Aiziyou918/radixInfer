from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from radixinfer.cache.page_pool import KVCacheView


@dataclass(frozen=True)
class MaterializedBatchMetadata:
    positions: tuple[int, ...] = field(default_factory=tuple)
    input_table_slots: tuple[int, ...] = field(default_factory=tuple)
    input_positions: tuple[int, ...] = field(default_factory=tuple)
    write_table_slots: tuple[int, ...] = field(default_factory=tuple)
    write_positions: tuple[int, ...] = field(default_factory=tuple)
    request_token_counts: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AttentionCacheWrite:
    keys: object
    values: object
    token_count: int


@dataclass(frozen=True)
class PrefillInput:
    request_ids: list[int]
    token_ids: list[list[int]]
    kv_caches: list[KVCacheView | None] = field(default_factory=list)
    metadata: MaterializedBatchMetadata | None = None


@dataclass(frozen=True)
class PrefillOutput:
    kv_writes: list[AttentionCacheWrite] = field(default_factory=list)


@dataclass(frozen=True)
class DecodeInput:
    request_ids: list[int]
    token_ids: list[list[int]]
    kv_caches: list[KVCacheView | None] = field(default_factory=list)
    metadata: MaterializedBatchMetadata | None = None


@dataclass(frozen=True)
class DecodeOutput:
    next_token_ids: list[int]
    kv_writes: list[AttentionCacheWrite] = field(default_factory=list)


class Engine(Protocol):
    def prefill(self, batch: PrefillInput) -> PrefillOutput:
        ...

    def decode(self, batch: DecodeInput) -> DecodeOutput:
        ...
