from __future__ import annotations

from dataclasses import dataclass

from radixinfer.cache.page_pool import PagePool
from radixinfer.engine.base import DecodeInput

from .table import TableManager
from .types import RuntimeRequest


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

    def prepare_decode_input(self, requests: list[RuntimeRequest]) -> DecodeInput:
        return DecodeInput(
            request_ids=[req.request_id for req in requests],
            token_ids=[self.page_pool.read_span(req.cache_span)[-1:] for req in requests],  # type: ignore[arg-type]
            kv_caches=[
                self.page_pool.read_kv(req.cache_span, token_count=max(0, req.cached_token_count - 1))
                for req in requests
            ],  # type: ignore[arg-type]
        )
