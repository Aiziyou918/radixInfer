from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .page_pool import PageSpan


def _normalize_prefix(tokens: list[int], page_size: int) -> tuple[int, ...]:
    prefix_len = len(tokens) - (len(tokens) % page_size)
    return tuple(tokens[:prefix_len])


@dataclass
class PrefixHit:
    matched_tokens: int
    cached_span: PageSpan | None = None


class PrefixStore:
    def __init__(self, capacity: int, page_size: int) -> None:
        self.capacity = capacity
        self.page_size = page_size
        self._entries: OrderedDict[tuple[int, ...], PageSpan] = OrderedDict()

    def match(self, tokens: list[int]) -> PrefixHit:
        best = 0
        best_span = None
        for key, span in self._entries.items():
            if len(key) <= best:
                continue
            if len(tokens) >= len(key) and tuple(tokens[: len(key)]) == key:
                best = len(key)
                best_span = span
        return PrefixHit(matched_tokens=best, cached_span=best_span)

    def insert(self, tokens: list[int], span: PageSpan) -> None:
        key = _normalize_prefix(tokens, self.page_size)
        if not key:
            return
        self._entries.pop(key, None)
        self._entries[key] = span
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
