from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TableManager:
    max_running_requests: int
    page_size: int
    max_tokens_per_request: int
    _free_slots: list[int] = field(init=False)
    page_table: list[list[int | None]] = field(init=False)
    token_table: list[list[int | None]] = field(init=False)

    def __post_init__(self) -> None:
        self._free_slots = list(range(self.max_running_requests))
        self.page_table = [
            [None for _ in range(self.max_tokens_per_request)] for _ in range(self.max_running_requests)
        ]
        self.token_table = [
            [None for _ in range(self.max_tokens_per_request)] for _ in range(self.max_running_requests)
        ]

    @property
    def available_slots(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int | None:
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self.page_table[slot] = [None for _ in range(self.max_tokens_per_request)]
        self.token_table[slot] = [None for _ in range(self.max_tokens_per_request)]
        self._free_slots.append(slot)

    def materialize_span(self, slot: int, page_ids: tuple[int, ...], token_ids: list[int]) -> None:
        for index, token_id in enumerate(token_ids):
            page_index = index // self.page_size
            self.token_table[slot][index] = token_id
            self.page_table[slot][index] = page_ids[page_index]

    def append_token(self, slot: int, position: int, page_id: int, token_id: int) -> None:
        self.token_table[slot][position] = token_id
        self.page_table[slot][position] = page_id
