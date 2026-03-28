from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageReservation:
    page_ids: list[int]


@dataclass
class PagePool:
    total_pages: int
    page_size: int
    _free_pages: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._free_pages = list(range(self.total_pages))

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    def reserve_for_tokens(self, token_count: int) -> PageReservation | None:
        needed = max(1, (token_count + self.page_size - 1) // self.page_size)
        if needed > len(self._free_pages):
            return None
        page_ids = self._free_pages[:needed]
        del self._free_pages[:needed]
        return PageReservation(page_ids=page_ids)

    def release(self, reservation: PageReservation) -> None:
        self._free_pages.extend(reservation.page_ids)
        self._free_pages.sort()
