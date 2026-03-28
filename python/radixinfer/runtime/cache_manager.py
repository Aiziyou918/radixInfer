from __future__ import annotations

from dataclasses import dataclass

from radixinfer.cache.page_pool import PagePool, PageReservation, PageSpan
from radixinfer.cache.prefix_store import PrefixCacheKey, PrefixStore


@dataclass
class CacheManager:
    page_pool: PagePool
    prefix_store: PrefixStore
    page_size: int

    @property
    def available_tokens(self) -> int:
        return self.page_pool.free_pages * self.page_size + self.prefix_store.size_info.evictable_size

    def lock_prefix(self, key: PrefixCacheKey | None) -> None:
        self.prefix_store.lock(key)

    def unlock_prefix(self, key: PrefixCacheKey | None) -> None:
        self.prefix_store.unlock(key)

    def reserve(self, token_count: int, prefix_span: PageSpan | None) -> PageReservation | None:
        reservation = self.page_pool.reserve_for_tokens(token_count, prefix_span=prefix_span)
        if reservation is not None:
            return reservation
        needed_private_pages = self.page_pool.required_private_pages(token_count, prefix_span=prefix_span)
        missing_pages = max(0, needed_private_pages - self.page_pool.free_pages)
        if missing_pages == 0:
            return None
        missing_tokens = missing_pages * self.page_size
        if self.prefix_store.size_info.evictable_size < missing_tokens:
            return None
        for span in self.prefix_store.evict(missing_tokens):
            self.page_pool.evict_shared(span)
        return self.page_pool.reserve_for_tokens(token_count, prefix_span=prefix_span)

    def commit_prefix(self, tokens: list[int], span: PageSpan) -> PrefixCacheKey | None:
        self.page_pool.share_span(span)
        new_key, evicted_spans = self.prefix_store.insert(tokens, span)
        for evicted_span in evicted_spans:
            self.page_pool.evict_shared(evicted_span)
        return new_key

    def release(self, reservation: PageReservation | None) -> None:
        if reservation is not None:
            self.page_pool.release(reservation)
