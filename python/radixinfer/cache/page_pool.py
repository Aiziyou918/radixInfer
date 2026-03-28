from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PageSpan:
    page_ids: tuple[int, ...]
    token_count: int


@dataclass(frozen=True)
class KVCacheView:
    keys: torch.Tensor
    values: torch.Tensor
    token_count: int


@dataclass
class PageReservation:
    private_page_ids: list[int]
    token_count: int
    shared_page_ids: tuple[int, ...] = ()
    committed_tokens: int = 0

    @property
    def capacity_tokens(self) -> int:
        return len(self.page_ids) * self.token_count

    @property
    def page_ids(self) -> list[int]:
        return list(self.shared_page_ids) + self.private_page_ids


@dataclass
class PagePool:
    total_pages: int
    page_size: int
    kv_cache_dim: int = 16
    kv_num_layers: int = 2
    kv_num_heads: int = 2
    _free_pages: list[int] = field(init=False)
    _page_data: dict[int, list[int]] = field(init=False)
    _shared_page_refcounts: dict[int, int] = field(init=False)
    _key_cache: torch.Tensor = field(init=False)
    _value_cache: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self._free_pages = list(range(self.total_pages))
        self._page_data = {page_id: [] for page_id in range(self.total_pages)}
        self._shared_page_refcounts = {page_id: 0 for page_id in range(self.total_pages)}
        self._key_cache = torch.zeros(
            (
                self.kv_num_layers,
                self.total_pages,
                self.page_size,
                self.kv_num_heads,
                self.kv_cache_dim,
            ),
            dtype=torch.float32,
        )
        self._value_cache = torch.zeros_like(self._key_cache)

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    def reserve_for_tokens(
        self,
        token_count: int,
        *,
        prefix_span: PageSpan | None = None,
    ) -> PageReservation | None:
        needed = max(1, (token_count + self.page_size - 1) // self.page_size)
        shared_page_ids = prefix_span.page_ids if prefix_span is not None else ()
        private_needed = max(0, needed - len(shared_page_ids))
        if private_needed > len(self._free_pages):
            return None
        private_page_ids = self._free_pages[:private_needed]
        del self._free_pages[:private_needed]
        return PageReservation(
            private_page_ids=private_page_ids,
            shared_page_ids=shared_page_ids,
            token_count=self.page_size,
        )

    def write_tokens(
        self,
        reservation: PageReservation,
        tokens: list[int],
        *,
        start_offset: int = 0,
    ) -> PageSpan:
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        end_offset = start_offset + len(tokens)
        if end_offset > reservation.capacity_tokens:
            raise ValueError("token write exceeds reservation capacity")

        remaining = list(tokens)
        page_index = start_offset // self.page_size
        in_page_offset = start_offset % self.page_size
        while remaining:
            page_id = reservation.page_ids[page_index]
            page_tokens = self._page_data[page_id]
            while len(page_tokens) < in_page_offset:
                page_tokens.append(0)
            write_len = min(self.page_size - in_page_offset, len(remaining))
            chunk = remaining[:write_len]
            remaining = remaining[write_len:]
            for idx, token in enumerate(chunk, start=in_page_offset):
                if idx < len(page_tokens):
                    page_tokens[idx] = token
                else:
                    page_tokens.append(token)
            page_index += 1
            in_page_offset = 0

        reservation.committed_tokens = max(reservation.committed_tokens, end_offset)
        used_pages = (reservation.committed_tokens + self.page_size - 1) // self.page_size
        return PageSpan(page_ids=tuple(reservation.page_ids[:used_pages]), token_count=reservation.committed_tokens)

    def read_span(self, span: PageSpan) -> list[int]:
        tokens: list[int] = []
        for page_id in span.page_ids:
            tokens.extend(self._page_data[page_id])
        return tokens[: span.token_count]

    def read_kv(self, span: PageSpan, token_count: int | None = None) -> KVCacheView:
        effective_count = span.token_count if token_count is None else min(token_count, span.token_count)
        used_pages = len(span.page_ids)
        page_index = list(span.page_ids)
        keys = self._key_cache[:, page_index].reshape(
            self.kv_num_layers,
            used_pages * self.page_size,
            self.kv_num_heads,
            self.kv_cache_dim,
        )
        values = self._value_cache[:, page_index].reshape(
            self.kv_num_layers,
            used_pages * self.page_size,
            self.kv_num_heads,
            self.kv_cache_dim,
        )
        return KVCacheView(
            keys=keys[:, :effective_count].clone(),
            values=values[:, :effective_count].clone(),
            token_count=effective_count,
        )

    def write_kv(
        self,
        reservation: PageReservation,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        start_offset: int = 0,
    ) -> PageSpan:
        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape")
        if keys.ndim != 4:
            raise ValueError("expected KV tensors with shape (layers, tokens, heads, dim)")
        _, token_count, _, _ = keys.shape
        end_offset = start_offset + token_count
        if end_offset > reservation.capacity_tokens:
            raise ValueError("KV write exceeds reservation capacity")
        page_index = start_offset // self.page_size
        in_page_offset = start_offset % self.page_size
        remaining = token_count
        src_offset = 0
        layer_count = min(self.kv_num_layers, keys.shape[0])
        head_count = min(self.kv_num_heads, keys.shape[2])
        dim_count = min(self.kv_cache_dim, keys.shape[3])
        while remaining > 0:
            page_id = reservation.page_ids[page_index]
            write_len = min(self.page_size - in_page_offset, remaining)
            src_slice = slice(src_offset, src_offset + write_len)
            dst_slice = slice(in_page_offset, in_page_offset + write_len)
            self._key_cache[:, page_id, dst_slice].zero_()
            self._value_cache[:, page_id, dst_slice].zero_()
            self._key_cache[:layer_count, page_id, dst_slice, :head_count, :dim_count] = keys[
                :layer_count, src_slice, :head_count, :dim_count
            ].to(dtype=torch.float32)
            self._value_cache[:layer_count, page_id, dst_slice, :head_count, :dim_count] = values[
                :layer_count, src_slice, :head_count, :dim_count
            ].to(dtype=torch.float32)
            page_index += 1
            in_page_offset = 0
            src_offset += write_len
            remaining -= write_len

        reservation.committed_tokens = max(reservation.committed_tokens, end_offset)
        used_pages = (reservation.committed_tokens + self.page_size - 1) // self.page_size
        return PageSpan(page_ids=tuple(reservation.page_ids[:used_pages]), token_count=reservation.committed_tokens)

    def share_span(self, span: PageSpan) -> None:
        for page_id in span.page_ids:
            self._shared_page_refcounts[page_id] += 1

    def evict_shared(self, span: PageSpan) -> None:
        for page_id in span.page_ids:
            refs = self._shared_page_refcounts[page_id]
            if refs <= 0:
                raise ValueError(f"page {page_id} is not shared")
            refs -= 1
            self._shared_page_refcounts[page_id] = refs
            if refs == 0 and page_id not in self._free_pages:
                self._page_data[page_id] = []
                self._key_cache[:, page_id].zero_()
                self._value_cache[:, page_id].zero_()
                self._free_pages.append(page_id)
        self._free_pages.sort()

    def release(self, reservation: PageReservation) -> None:
        for page_id in reservation.private_page_ids:
            if self._shared_page_refcounts[page_id] > 0:
                continue
            self._page_data[page_id] = []
            self._key_cache[:, page_id].zero_()
            self._value_cache[:, page_id].zero_()
            if page_id not in self._free_pages:
                self._free_pages.append(page_id)
        self._free_pages.sort()

    def shared_refcount(self, page_id: int) -> int:
        return self._shared_page_refcounts[page_id]
