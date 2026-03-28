from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


def _normalize_prefix(tokens: list[int], page_size: int) -> tuple[int, ...]:
    prefix_len = len(tokens) - (len(tokens) % page_size)
    return tuple(tokens[:prefix_len])


@dataclass
class PrefixHit:
    matched_tokens: int


class PrefixStore:
    def __init__(self, capacity: int, page_size: int) -> None:
        self.capacity = capacity
        self.page_size = page_size
        self._entries: OrderedDict[tuple[int, ...], int] = OrderedDict()

    def match(self, tokens: list[int]) -> PrefixHit:
        best = 0
        for key in self._entries.keys():
            if len(key) <= best:
                continue
            if len(tokens) >= len(key) and tuple(tokens[: len(key)]) == key:
                best = len(key)
        return PrefixHit(matched_tokens=best)

    def insert(self, tokens: list[int]) -> None:
        key = _normalize_prefix(tokens, self.page_size)
        if not key:
            return
        self._entries.pop(key, None)
        self._entries[key] = len(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
