from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from radixinfer.cache.page_pool import PageReservation, PageSpan
from radixinfer.cache.prefix_store import PrefixCacheKey
from radixinfer.transport.protocol import SamplingParams


class RequestPhase(str, Enum):
    WAIT_PREFILL = "wait_prefill"
    PREFILLING = "prefilling"
    READY_TO_DECODE = "ready_to_decode"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class RuntimeRequest:
    request_id: int
    prompt_tokens: list[int]
    sampling: SamplingParams
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    phase: RequestPhase = RequestPhase.WAIT_PREFILL
    generated_tokens: list[int] = field(default_factory=list)
    prefix_matched: int = 0
    prefix_span: PageSpan | None = None
    prefix_cache_key: PrefixCacheKey | None = None
    cache_span: PageSpan | None = None
    prefill_cursor: int = 0
    age: int = 0
    reserved_tokens: int = 0
    table_slot: int | None = None
    reservation: PageReservation | None = None

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(0, len(self.prompt_tokens) - self.prefix_matched)

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.uncached_prompt_tokens - self.prefill_cursor)

    @property
    def prefill_complete(self) -> bool:
        return self.remaining_prefill_tokens == 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.sampling.max_tokens - len(self.generated_tokens))

    @property
    def cached_token_count(self) -> int:
        return 0 if self.cache_span is None else self.cache_span.token_count

    @property
    def finished(self) -> bool:
        return self.phase in {RequestPhase.FINISHED, RequestPhase.ABORTED}
