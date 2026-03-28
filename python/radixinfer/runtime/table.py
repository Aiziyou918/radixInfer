from __future__ import annotations

import torch


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
