from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from radixinfer.cache.page_pool import PageReservation
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
    phase: RequestPhase = RequestPhase.WAIT_PREFILL
    generated_tokens: list[int] = field(default_factory=list)
    prefix_matched: int = 0
    age: int = 0
    reservation: PageReservation | None = None

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.sampling.max_tokens - len(self.generated_tokens))

    @property
    def finished(self) -> bool:
        return self.phase in {RequestPhase.FINISHED, RequestPhase.ABORTED}
