from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageSpan:
    page_ids: tuple[int, ...]
    token_count: int


@dataclass
class PageReservation:
    page_ids: list[int]
    token_count: int
    committed_tokens: int = 0

    @property
    def capacity_tokens(self) -> int:
        return len(self.page_ids) * self.token_count


@dataclass
class PagePool:
    total_pages: int
    page_size: int
    _free_pages: list[int] = field(init=False)
    _page_data: dict[int, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self._free_pages = list(range(self.total_pages))
        self._page_data = {page_id: [] for page_id in range(self.total_pages)}

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    def reserve_for_tokens(self, token_count: int) -> PageReservation | None:
        needed = max(1, (token_count + self.page_size - 1) // self.page_size)
        if needed > len(self._free_pages):
            return None
        page_ids = self._free_pages[:needed]
        del self._free_pages[:needed]
        return PageReservation(page_ids=page_ids, token_count=self.page_size)

    def write_tokens(
        self,
        reservation: PageReservation,
        tokens: list[int],
        *,
        start_offset: int = 0,
    ) -> PageSpan:
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        end_offset = start_offset + len(tokens)
        if end_offset > reservation.capacity_tokens:
            raise ValueError("token write exceeds reservation capacity")

        remaining = list(tokens)
        page_index = start_offset // self.page_size
        in_page_offset = start_offset % self.page_size
        while remaining:
            page_id = reservation.page_ids[page_index]
            page_tokens = self._page_data[page_id]
            while len(page_tokens) < in_page_offset:
                page_tokens.append(0)
            write_len = min(self.page_size - in_page_offset, len(remaining))
            chunk = remaining[:write_len]
            remaining = remaining[write_len:]
            for idx, token in enumerate(chunk, start=in_page_offset):
                if idx < len(page_tokens):
                    page_tokens[idx] = token
                else:
                    page_tokens.append(token)
            page_index += 1
            in_page_offset = 0

        reservation.committed_tokens = max(reservation.committed_tokens, end_offset)
        used_pages = (reservation.committed_tokens + self.page_size - 1) // self.page_size
        return PageSpan(page_ids=tuple(reservation.page_ids[:used_pages]), token_count=reservation.committed_tokens)

    def read_span(self, span: PageSpan) -> list[int]:
        tokens: list[int] = []
        for page_id in span.page_ids:
            tokens.extend(self._page_data[page_id])
        return tokens[: span.token_count]

    def release(self, reservation: PageReservation) -> None:
        for page_id in reservation.page_ids:
            self._page_data[page_id] = []
        self._free_pages.extend(reservation.page_ids)
        self._free_pages.sort()
