from __future__ import annotations

import torch
from radixinfer.engine.base import RequestPagedAttentionState, RequestTableState


class TableManager:
    """Manages per-request table slots and the shared token_pool/page_table tensors.

    token_pool: GPU tensor of shape (max_running_reqs+1, aligned_max_seq_len), dtype int32.
    Mirrors the global page_table layout so that input_ids can be gathered in one operation.
    """

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # token_pool has the same shape as page_table; dummy req uses slot max_running_reqs
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)


class _DebugTableManager:
    """CPU-only table manager used by SchedulerRuntime debug mode.

    Tracks token_table and page_table as plain Python dicts of lists so that
    tests can inspect them directly without GPU tensors.
    """

    def __init__(self, max_slots: int, page_size: int) -> None:
        self.page_size = page_size
        self._free_slots = list(range(max_slots))
        self.token_table: dict[int, list[int]] = {}
        self.page_table: dict[int, list[int]] = {}

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        slot = self._free_slots.pop()
        self.token_table[slot] = []
        self.page_table[slot] = []
        return slot

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
        self.token_table.pop(slot, None)
        self.page_table.pop(slot, None)

    def materialize_span(self, slot: int, page_ids: tuple, token_ids: list) -> None:
        pages_per_pos = [page_ids[i // self.page_size] for i in range(len(token_ids))]
        self.token_table[slot] = list(token_ids)
        self.page_table[slot] = pages_per_pos

    def append_token(self, slot: int, position: int, page_id: int, token_id: int) -> None:
        tokens = self.token_table.setdefault(slot, [])
        pages = self.page_table.setdefault(slot, [])
        while len(tokens) <= position:
            tokens.append(0)
            pages.append(0)
        tokens[position] = token_id
        pages[position] = page_id

    def request_state(self, slot: int, token_count: int, write_position: int) -> RequestTableState:
        tokens = self.token_table.get(slot, [])
        pages = self.page_table.get(slot, [])
        return RequestTableState(
            table_slot=slot,
            token_count=token_count,
            write_position=write_position,
            page_ids=tuple(pages[:token_count]),
            token_ids=tuple(tokens[:token_count]),
        )

    def paged_attention_state(
        self, slot: int, token_count: int, write_position: int
    ) -> RequestPagedAttentionState:
        pages = self.page_table.get(slot, [])
        pages_up_to = pages[:token_count]
        seen: dict[int, int] = {}
        unique_pages: list[int] = []
        for p in pages_up_to:
            if p not in seen:
                seen[p] = len(unique_pages)
                unique_pages.append(p)
        last_page_len = token_count - (len(unique_pages) - 1) * self.page_size if unique_pages else 0
        return RequestPagedAttentionState(
            table_slot=slot,
            token_count=token_count,
            write_position=write_position,
            page_ids=tuple(unique_pages),
            page_indices=tuple(unique_pages),
            kv_page_indices=tuple(unique_pages),
            last_page_len=last_page_len,
        )
