from __future__ import annotations

from dataclasses import dataclass

from radixinfer.cache.page_pool import PagePool
from radixinfer.engine.base import DecodeInput

from .table import TableManager
from .types import RuntimeRequest


@dataclass(frozen=True)
class BatchMetadata:
    positions: list[int]
    input_table_slots: list[int]
    input_positions: list[int]
    write_table_slots: list[int]
    write_positions: list[int]


@dataclass(frozen=True)
class PreparedDecodeBatch:
    decode_input: DecodeInput
    metadata: BatchMetadata


@dataclass
class Executor:
    page_pool: PagePool
    table_manager: TableManager

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

    def prepare_decode_batch(self, requests: list[RuntimeRequest]) -> PreparedDecodeBatch:
        return PreparedDecodeBatch(
            decode_input=DecodeInput(
                request_ids=[req.request_id for req in requests],
                token_ids=[self.page_pool.read_span(req.cache_span)[-1:] for req in requests],  # type: ignore[arg-type]
                kv_caches=[
                    self.page_pool.read_kv(req.cache_span, token_count=max(0, req.cached_token_count - 1))
                    for req in requests
                ],  # type: ignore[arg-type]
            ),
            metadata=BatchMetadata(
                positions=self._make_positions(requests),
                input_table_slots=self._make_input_table_slots(requests),
                input_positions=self._make_input_positions(requests),
                write_table_slots=self._make_write_table_slots(requests),
                write_positions=self._make_write_positions(requests),
            ),
        )

    def _make_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count - 1 for req in requests]

    def _make_input_table_slots(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.table_slot for req in requests if req.table_slot is not None]

    def _make_input_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count - 1 for req in requests]

    def _make_write_table_slots(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.table_slot for req in requests if req.table_slot is not None]

    def _make_write_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count for req in requests]
