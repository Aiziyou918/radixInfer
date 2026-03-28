from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DecodeInput:
    request_ids: list[int]
    token_ids: list[list[int]]


@dataclass(frozen=True)
class DecodeOutput:
    next_token_ids: list[int]


class Engine(Protocol):
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        ...
