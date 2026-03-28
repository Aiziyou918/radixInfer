from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from .page_pool import PageSpan


def _normalize_prefix(tokens: list[int], page_size: int) -> tuple[int, ...]:
    prefix_len = len(tokens) - (len(tokens) % page_size)
    return tuple(tokens[:prefix_len])


@dataclass(frozen=True)
class PrefixCacheKey:
    tokens: tuple[int, ...]


@dataclass
class PrefixHit:
    matched_tokens: int
    cached_span: PageSpan | None = None
    cache_key: PrefixCacheKey | None = None


@dataclass
class PrefixEntry:
    key: PrefixCacheKey
    span: PageSpan
    ref_count: int = 0
    last_access_ns: int = 0

    @property
    def evictable(self) -> bool:
        return self.ref_count == 0


class PrefixStore:
    def __init__(self, capacity: int, page_size: int) -> None:
        self.capacity = capacity
        self.page_size = page_size
        self._entries: OrderedDict[PrefixCacheKey, PrefixEntry] = OrderedDict()

    def match(self, tokens: list[int]) -> PrefixHit:
        best_key: PrefixCacheKey | None = None
        best_entry: PrefixEntry | None = None
        for key, entry in self._entries.items():
            if best_key is not None and len(key.tokens) <= len(best_key.tokens):
                continue
            if len(tokens) >= len(key.tokens) and tuple(tokens[: len(key.tokens)]) == key.tokens:
                best_key = key
                best_entry = entry
        if best_entry is None or best_key is None:
            return PrefixHit(matched_tokens=0)
        best_entry.last_access_ns = time.monotonic_ns()
        self._entries.move_to_end(best_key)
        return PrefixHit(
            matched_tokens=len(best_key.tokens),
            cached_span=best_entry.span,
            cache_key=best_key,
        )

    def lock(self, key: PrefixCacheKey | None) -> None:
        if key is None:
            return
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.ref_count += 1
        entry.last_access_ns = time.monotonic_ns()
        self._entries.move_to_end(key)

    def unlock(self, key: PrefixCacheKey | None) -> None:
        if key is None:
            return
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.ref_count <= 0:
            raise ValueError("prefix entry ref_count underflow")
        entry.ref_count -= 1
        entry.last_access_ns = time.monotonic_ns()

    def insert(self, tokens: list[int], span: PageSpan) -> tuple[PrefixCacheKey | None, list[PageSpan]]:
        key_tokens = _normalize_prefix(tokens, self.page_size)
        if not key_tokens:
            return None, []
        key = PrefixCacheKey(key_tokens)
        now = time.monotonic_ns()
        evicted: list[PageSpan] = []
        old_entry = self._entries.pop(key, None)
        ref_count = 0
        if old_entry is not None:
            ref_count = old_entry.ref_count
            evicted.append(old_entry.span)
        self._entries[key] = PrefixEntry(
            key=key,
            span=span,
            ref_count=ref_count,
            last_access_ns=now,
        )
        evicted.extend(self._evict_over_capacity())
        return key, evicted

    def _evict_over_capacity(self) -> list[PageSpan]:
        evicted: list[PageSpan] = []
        while len(self._entries) > self.capacity:
            candidate_key = self._find_evictable_key()
            if candidate_key is None:
                break
            evicted.append(self._entries.pop(candidate_key).span)
        return evicted

    def _find_evictable_key(self) -> PrefixCacheKey | None:
        best_key: PrefixCacheKey | None = None
        best_time: int | None = None
        for key, entry in self._entries.items():
            if not entry.evictable:
                continue
            if best_time is None or entry.last_access_ns < best_time:
                best_key = key
                best_time = entry.last_access_ns
        return best_key

    def entry_ref_count(self, key: PrefixCacheKey) -> int:
        return self._entries[key].ref_count
