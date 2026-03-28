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

    def prepare_prefill_batch(self, requests: list[RuntimeRequest]) -> PreparedPrefillBatch:
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
        metadata = MaterializedBatchMetadata(
            positions=self._make_prefill_positions(requests),
            input_table_slots=self._make_table_slots(requests),
            input_positions=self._make_prefill_input_positions(requests),
            write_table_slots=self._make_table_slots(requests),
            write_positions=self._make_prefill_write_positions(requests),
        )
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
        metadata = MaterializedBatchMetadata(
            positions=self._make_decode_positions(requests),
            input_table_slots=self._make_table_slots(requests),
            input_positions=self._make_decode_input_positions(requests),
            write_table_slots=self._make_table_slots(requests),
            write_positions=self._make_decode_write_positions(requests),
        )
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

    def _make_table_slots(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.table_slot for req in requests if req.table_slot is not None]

    def _make_prefill_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        positions: list[int] = []
        for req in requests:
            positions.extend(range(req.prefix_matched, req.prefix_matched + req.prefill_cursor))
        return positions

    def _make_prefill_input_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return self._make_prefill_positions(requests)

    def _make_prefill_write_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        positions: list[int] = []
        for req in requests:
            positions.extend(range(req.prefix_matched, req.prefix_matched + req.prefill_cursor))
        return positions

    def _make_decode_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count - 1 for req in requests]

    def _make_decode_input_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count - 1 for req in requests]

    def _make_decode_write_positions(self, requests: list[RuntimeRequest]) -> list[int]:
        return [req.cached_token_count for req in requests]
