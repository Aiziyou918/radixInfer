from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from radixinfer.cache.page_pool import KVCacheView


@dataclass(frozen=True)
class RequestTableState:
    table_slot: int
    token_count: int
    write_position: int
    page_ids: tuple[int | None, ...] = field(default_factory=tuple)
    token_ids: tuple[int | None, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RequestPagedAttentionState:
    table_slot: int
    token_count: int
    write_position: int
    page_ids: tuple[int, ...] = field(default_factory=tuple)
    page_indices: tuple[int, ...] = field(default_factory=tuple)
    kv_page_indices: tuple[int, ...] = field(default_factory=tuple)
    last_page_len: int = 0


@dataclass(frozen=True)
class MaterializedBatchMetadata:
    positions: tuple[int, ...] = field(default_factory=tuple)
    input_table_slots: tuple[int, ...] = field(default_factory=tuple)
    input_positions: tuple[int, ...] = field(default_factory=tuple)
    write_table_slots: tuple[int, ...] = field(default_factory=tuple)
    write_positions: tuple[int, ...] = field(default_factory=tuple)
    request_token_counts: tuple[int, ...] = field(default_factory=tuple)
    request_table_states: tuple[RequestTableState, ...] = field(default_factory=tuple)
    request_paged_states: tuple[RequestPagedAttentionState, ...] = field(default_factory=tuple)

    def request_slice(self, index: int) -> slice:
        start = sum(self.request_token_counts[:index])
        end = start + self.request_token_counts[index]
        return slice(start, end)

    def request_view(self, index: int) -> "MaterializedBatchMetadata":
        request_slice = self.request_slice(index)
        return MaterializedBatchMetadata(
            positions=self.positions[request_slice],
            input_table_slots=self.input_table_slots[request_slice],
            input_positions=self.input_positions[request_slice],
            write_table_slots=self.write_table_slots[index : index + 1],
            write_positions=self.write_positions[index : index + 1],
            request_token_counts=self.request_token_counts[index : index + 1],
            request_table_states=self.request_table_states[index : index + 1],
            request_paged_states=self.request_paged_states[index : index + 1],
        )


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
