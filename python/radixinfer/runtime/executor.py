from __future__ import annotations

from dataclasses import dataclass

from radixinfer.cache.page_pool import PagePool
from radixinfer.engine.base import DecodeInput, MaterializedBatchMetadata, PrefillInput

from .table import TableManager
from .types import RuntimeRequest


@dataclass(frozen=True)
class PreparedPrefillBatch:
    prefill_input: PrefillInput
    metadata: MaterializedBatchMetadata


@dataclass(frozen=True)
class PreparedDecodeBatch:
    decode_input: DecodeInput
    metadata: MaterializedBatchMetadata


@dataclass
class _BatchMetadataBuffer:
    positions: list[int]
    input_table_slots: list[int]
    input_positions: list[int]
    write_table_slots: list[int]
    write_positions: list[int]
    request_token_counts: list[int]

    def reset(self) -> None:
        self.positions.clear()
        self.input_table_slots.clear()
        self.input_positions.clear()
        self.write_table_slots.clear()
        self.write_positions.clear()
        self.request_token_counts.clear()


@dataclass
class Executor:
    page_pool: PagePool
    table_manager: TableManager

    def __post_init__(self) -> None:
        self._prefill_buffer = _BatchMetadataBuffer([], [], [], [], [], [])
        self._decode_buffer = _BatchMetadataBuffer([], [], [], [], [], [])

    def materialize_request(self, request: RuntimeRequest) -> None:
        if request.table_slot is None or request.cache_span is None:
            return
        token_ids = self.page_pool.read_span(request.cache_span)
        self.table_manager.materialize_span(request.table_slot, request.cache_span.page_ids, token_ids)

    def append_token(self, request: RuntimeRequest, token_id: int) -> None:
        if request.table_slot is None or request.cache_span is None:
            return
        position = request.cached_token_count - 1
        page_id = request.cache_span.page_ids[position // self.table_manager.page_size]
        self.table_manager.append_token(request.table_slot, position, page_id, token_id)

    def prepare_prefill_batch(self, requests: list[RuntimeRequest]) -> PreparedPrefillBatch:
        for req in requests:
            if req.table_slot is not None and req.cache_span is not None and req.cached_token_count > 0:
                self.materialize_request(req)
        metadata = self._build_prefill_metadata(requests)
        token_ids = [
            req.prompt_tokens[req.prefix_matched : req.prefix_matched + req.prefill_cursor]
            for req in requests
        ]
        kv_caches = [
            (
                self.page_pool.read_kv(req.cache_span, token_count=req.prefix_matched)
                if req.cache_span is not None and req.prefix_matched > 0
                else None
            )
            for req in requests
        ]
        return PreparedPrefillBatch(
            prefill_input=PrefillInput(
                request_ids=[req.request_id for req in requests],
                token_ids=token_ids,
                kv_caches=kv_caches,
                metadata=metadata,
            ),
            metadata=metadata,
        )

    def prepare_decode_batch(self, requests: list[RuntimeRequest]) -> PreparedDecodeBatch:
        for req in requests:
            if req.table_slot is not None and req.cache_span is not None and req.cached_token_count > 0:
                self.materialize_request(req)
        metadata = self._build_decode_metadata(requests)
        return PreparedDecodeBatch(
            decode_input=DecodeInput(
                request_ids=[req.request_id for req in requests],
                token_ids=[self.page_pool.read_span(req.cache_span)[-1:] for req in requests],  # type: ignore[arg-type]
                kv_caches=[
                    self.page_pool.read_kv(req.cache_span, token_count=max(0, req.cached_token_count - 1))
                    for req in requests
                ],  # type: ignore[arg-type]
                metadata=metadata,
            ),
            metadata=metadata,
        )

    def _build_prefill_metadata(self, requests: list[RuntimeRequest]) -> MaterializedBatchMetadata:
        buffer = self._prefill_buffer
        buffer.reset()
        for req in requests:
            if req.table_slot is None:
                continue
            token_count = req.prefill_cursor
            start = req.prefix_matched
            stop = start + token_count
            buffer.positions.extend(range(start, stop))
            buffer.input_positions.extend(range(start, stop))
            buffer.input_table_slots.extend([req.table_slot] * token_count)
            buffer.write_table_slots.append(req.table_slot)
            buffer.write_positions.append(stop)
            buffer.request_token_counts.append(token_count)
        return MaterializedBatchMetadata(
            positions=tuple(buffer.positions),
            input_table_slots=tuple(buffer.input_table_slots),
            input_positions=tuple(buffer.input_positions),
            write_table_slots=tuple(buffer.write_table_slots),
            write_positions=tuple(buffer.write_positions),
            request_token_counts=tuple(buffer.request_token_counts),
            request_table_states=tuple(
                self.table_manager.request_state(
                    req.table_slot,
                    token_count=req.prefix_matched,
                    write_position=req.prefix_matched + req.prefill_cursor,
                )
                for req in requests
                if req.table_slot is not None
            ),
            request_paged_states=tuple(
                self.table_manager.paged_attention_state(
                    req.table_slot,
                    token_count=req.prefix_matched,
                    write_position=req.prefix_matched + req.prefill_cursor,
                )
                for req in requests
                if req.table_slot is not None
            ),
        )

    def _build_decode_metadata(self, requests: list[RuntimeRequest]) -> MaterializedBatchMetadata:
        buffer = self._decode_buffer
        buffer.reset()
        for req in requests:
            if req.table_slot is None:
                continue
            buffer.positions.append(req.cached_token_count - 1)
            buffer.input_table_slots.append(req.table_slot)
            buffer.input_positions.append(req.cached_token_count - 1)
            buffer.write_table_slots.append(req.table_slot)
            buffer.write_positions.append(req.cached_token_count)
            buffer.request_token_counts.append(1)
        return MaterializedBatchMetadata(
            positions=tuple(buffer.positions),
            input_table_slots=tuple(buffer.input_table_slots),
            input_positions=tuple(buffer.input_positions),
            write_table_slots=tuple(buffer.write_table_slots),
            write_positions=tuple(buffer.write_positions),
            request_token_counts=tuple(buffer.request_token_counts),
            request_table_states=tuple(
                self.table_manager.request_state(
                    req.table_slot,
                    token_count=req.cached_token_count,
                    write_position=req.cached_token_count,
                )
                for req in requests
                if req.table_slot is not None
            ),
            request_paged_states=tuple(
                self.table_manager.paged_attention_state(
                    req.table_slot,
                    token_count=req.cached_token_count,
                    write_position=req.cached_token_count,
                )
                for req in requests
                if req.table_slot is not None
            ),
        )
