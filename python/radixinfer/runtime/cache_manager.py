from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch

from radixinfer.core import Req
from radixinfer.utils import div_ceil

if TYPE_CHECKING:
    from radixinfer.runtime.utils import PendingReq


class CacheManager:
    """Manages KV cache page allocation, prefix matching, and eviction.

    Aligns with mini-sglang CacheManager interface:
    - free_slots: GPU tensor of page-aligned free slot indices
    - prefix_cache: radix/lru prefix cache (BasePrefixCache interface)
    - allocate_paged: bulk page allocation with H2D page_table writes
    - cache_req: insert prefix into cache after prefill
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
    ):
        device = page_table.device
        self.free_slots = (
            torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        )
        self.prefix_cache = _create_prefix_cache(device)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return (
            self.prefix_cache.size_info.evictable_size
            + len(self.free_slots) * self.page_size
        )

    def lock(self, handle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))

        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                f"CacheManager integrity check failed: "
                f"free_pages={len(self.free_slots)} + "
                f"cache_pages={cache_pages} != num_pages={self.num_pages}"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """Context manager that defers page frees to avoid list-cat overhead inside the loop."""
        lazy_free_list: List[torch.Tensor] = []

        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        original_free = self._free
        self._free = lazy_free
        try:
            yield
        finally:
            self._free = original_free
            if lazy_free_list:
                self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        free_pages = len(self.free_slots)
        if needed_pages > free_pages:
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos = first_page * page_size
        last_pos = last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets_gpu = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets_gpu] = allocated


def _create_prefix_cache(device: torch.device):
    from radixinfer.cache.prefix_store import RadixPrefixCache
    return RadixPrefixCache(device=device)
